import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score
import yaml
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import re
from utils import load_yaml_files

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze attention patterns in faithful vs unfaithful classification')
    parser.add_argument('-m', '--model_name', type=str, default='qwen2.5-1.5b-instruct', choices=['qwen2.5-1.5b-instruct', 'llama-3.1-8b-instruct'], help='Name of the model to analyze')
    parser.add_argument('-p', '--model_path', type=str, required=True, help='Path to the saved classifier model')
    parser.add_argument('-o', '--output_dir', type=str, default='figures/attention_analysis', help='Directory to save analysis results')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('-k', '--top_k', type=int, default=20, help='Number of top tokens to display')
    parser.add_argument('-n', '--num_examples', type=int, default=10, help='Number of examples to visualize')
    return parser.parse_args()

class TokenDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example['generated_cot']
        label = 1 if example['is_faithful'] else 0
        
        # Tokenize
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label),
            'example_id': example.get('example_id', str(idx)),
            'text': text
        }

class AttentionClassifier(torch.nn.Module):
    def __init__(self, model_path):
        super(AttentionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, 2)
        )
        
        # Load the saved weights
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Filter out classifier weights if they don't match
        filtered_state_dict = {}
        for name, param in state_dict.items():
            if name.startswith('bert.'):
                filtered_state_dict[name] = param
            elif name.startswith('classifier.'):
                # Check if the shape matches
                try:
                    self_param = self.state_dict()[name]
                    if self_param.shape == param.shape:
                        filtered_state_dict[name] = param
                    else:
                        print(f"Skipping parameter {name} due to shape mismatch: {self_param.shape} vs {param.shape}")
                except KeyError:
                    print(f"Parameter {name} not found in model")
        
        # Load the filtered state dict
        missing, unexpected = self.load_state_dict(filtered_state_dict, strict=False)
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        
        # Return logits and attention weights
        return logits, outputs.attentions

def analyze_attention_weights(model, dataloader, tokenizer, device, top_k=20, num_examples=10):
    """Analyze which tokens the CLS token attends to most."""
    model.eval()
    
    # Store attention scores for faithful and unfaithful examples
    faithful_attention = defaultdict(list)
    unfaithful_attention = defaultdict(list)
    
    # Store examples for visualization
    faithful_examples = []
    unfaithful_examples = []
    
    # Track predictions
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing attention"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits, attentions = model(input_ids, attention_mask)
            
            # Get predictions
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Get attention weights from the last layer
            # Shape: [batch_size, num_heads, seq_len, seq_len]
            last_layer_attention = attentions[-1]
            
            # Average across attention heads
            # Shape: [batch_size, seq_len, seq_len]
            avg_attention = last_layer_attention.mean(dim=1)
            
            # Get CLS token attention to all other tokens
            # Shape: [batch_size, seq_len]
            cls_attention = avg_attention[:, 0, :]
            
            # Process each example in the batch
            for i in range(input_ids.shape[0]):
                # Get tokens that are not padding
                mask = attention_mask[i].bool()
                
                # Get attention scores for non-padding tokens
                attn_scores = cls_attention[i, mask]
                token_ids = input_ids[i, mask]
                
                # Convert token IDs to tokens
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                
                # Create token-attention pairs
                token_attn_pairs = list(zip(tokens, attn_scores.cpu().numpy()))
                
                # Store by prediction (not ground truth)
                if preds[i] == 1:  # Predicted faithful
                    for token, score in token_attn_pairs:
                        faithful_attention[token].append(score)
                    
                    # Store example if we need more
                    if len(faithful_examples) < num_examples and labels[i] == 1:  # Correctly predicted
                        faithful_examples.append({
                            'text': batch['text'][i],
                            'tokens': tokens,
                            'attention': attn_scores.cpu().numpy(),
                            'example_id': batch['example_id'][i]
                        })
                else:  # Predicted unfaithful
                    for token, score in token_attn_pairs:
                        unfaithful_attention[token].append(score)
                    
                    # Store example if we need more
                    if len(unfaithful_examples) < num_examples and labels[i] == 0:  # Correctly predicted
                        unfaithful_examples.append({
                            'text': batch['text'][i],
                            'tokens': tokens,
                            'attention': attn_scores.cpu().numpy(),
                            'example_id': batch['example_id'][i]
                        })
    
    # Calculate average attention per token
    faithful_avg = {token: np.mean(scores) for token, scores in faithful_attention.items()}
    unfaithful_avg = {token: np.mean(scores) for token, scores in unfaithful_attention.items()}
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Classifier accuracy: {accuracy:.4f}")
    
    return {
        'faithful_avg': faithful_avg,
        'unfaithful_avg': unfaithful_avg,
        'faithful_examples': faithful_examples,
        'unfaithful_examples': unfaithful_examples
    }

def plot_top_tokens(results, output_dir, model_name, top_k=20):
    """Plot the top tokens by attention score."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top tokens for unfaithful examples
    top_unfaithful = sorted(results['unfaithful_avg'].items(), key=lambda x: x[1], reverse=True)
    
    # Filter out special tokens and punctuation
    filtered_unfaithful = [(token, score) for token, score in top_unfaithful 
                          if not token.startswith('[') and not token.endswith(']')
                          and not all(c in '.,;:!?-()[]{}"\'' for c in token)]
    
    # Get top tokens
    top_unfaithful = filtered_unfaithful[:top_k]
    
    # Plot
    plt.figure(figsize=(12, 8))
    tokens = [token for token, _ in top_unfaithful]
    scores = [score for _, score in top_unfaithful]
    
    plt.barh(tokens, scores, color='red', alpha=0.7)
    plt.title(f'Top {top_k} Tokens by CLS Attention in Unfaithful Classification')
    plt.xlabel('Average Attention Score')
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/{model_name}_top_unfaithful_tokens.png')
    plt.close()
    
    # Get top tokens for faithful examples
    top_faithful = sorted(results['faithful_avg'].items(), key=lambda x: x[1], reverse=True)
    
    # Filter out special tokens and punctuation
    filtered_faithful = [(token, score) for token, score in top_faithful 
                        if not token.startswith('[') and not token.endswith(']')
                        and not all(c in '.,;:!?-()[]{}"\'' for c in token)]
    
    # Get top tokens
    top_faithful = filtered_faithful[:top_k]
    
    # Plot
    plt.figure(figsize=(12, 8))
    tokens = [token for token, _ in top_faithful]
    scores = [score for _, score in top_faithful]
    
    plt.barh(tokens, scores, color='blue', alpha=0.7)
    plt.title(f'Top {top_k} Tokens by CLS Attention in Faithful Classification')
    plt.xlabel('Average Attention Score')
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/{model_name}_top_faithful_tokens.png')
    plt.close()
    
    # Compare attention between faithful and unfaithful for common tokens
    common_tokens = set(results['faithful_avg'].keys()) & set(results['unfaithful_avg'].keys())
    
    # Calculate ratio of unfaithful to faithful attention
    ratios = {}
    for token in common_tokens:
        faithful_score = results['faithful_avg'][token]
        unfaithful_score = results['unfaithful_avg'][token]
        
        if faithful_score > 0:
            ratios[token] = unfaithful_score / faithful_score
    
    # Get top tokens by ratio
    top_ratio = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
    
    # Filter out special tokens and punctuation
    filtered_ratio = [(token, ratio) for token, ratio in top_ratio 
                     if not token.startswith('[') and not token.endswith(']')
                     and not all(c in '.,;:!?-()[]{}"\'' for c in token)]
    
    # Get top tokens
    top_ratio = filtered_ratio[:top_k]
    
    # Plot
    plt.figure(figsize=(12, 8))
    tokens = [token for token, _ in top_ratio]
    ratios = [ratio for _, ratio in top_ratio]
    
    plt.barh(tokens, ratios, color='purple', alpha=0.7)
    plt.axvline(x=1, color='black', linestyle='--')
    plt.title(f'Top {top_k} Tokens by Unfaithful/Faithful Attention Ratio')
    plt.xlabel('Attention Ratio (Unfaithful / Faithful)')
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/{model_name}_attention_ratio.png')
    plt.close()
    
    # Create a dataframe for the top tokens by unfaithful attention
    df_unfaithful = pd.DataFrame({
        'Token': [token for token, _ in top_unfaithful],
        'Attention Score': [score for _, score in top_unfaithful]
    })
    
    # Save to CSV
    df_unfaithful.to_csv(f'{output_dir}/{model_name}_top_unfaithful_tokens.csv', index=False)
    
    # Create a dataframe for the top tokens by faithful attention
    df_faithful = pd.DataFrame({
        'Token': [token for token, _ in top_faithful],
        'Attention Score': [score for _, score in top_faithful]
    })
    
    # Save to CSV
    df_faithful.to_csv(f'{output_dir}/{model_name}_top_faithful_tokens.csv', index=False)
    
    # Create a dataframe for the top tokens by ratio
    df_ratio = pd.DataFrame({
        'Token': [token for token, _ in top_ratio],
        'Ratio': [ratio for _, ratio in top_ratio],
        'Unfaithful Score': [results['unfaithful_avg'][token] for token, _ in top_ratio],
        'Faithful Score': [results['faithful_avg'][token] for token, _ in top_ratio]
    })
    
    # Save to CSV
    df_ratio.to_csv(f'{output_dir}/{model_name}_attention_ratios.csv', index=False)
    
    return df_ratio

def visualize_example_attention(example, tokenizer, output_dir, model_name, example_id, is_faithful):
    """Visualize attention for a single example."""
    # Get text and attention scores
    text = example['text']
    tokens = example['tokens']
    attention = example['attention']
    
    # Create a more readable version of the text with token boundaries
    token_text = []
    for token in tokens:
        # Replace ## for subword tokens
        clean_token = token.replace('##', '')
        if token.startswith('##'):
            token_text.append(clean_token)
        else:
            token_text.append(' ' + clean_token)
    token_text = ''.join(token_text).strip()
    
    # Find correction markers
    correction_markers = [
        "actually", "in fact", "wait", "no,", "but", "however",
        "i was wrong", "i made a mistake", "i need to correct",
        "let me revise", "to clarify", "i meant", "i should have"
    ]
    
    highlighted_text = token_text
    for marker in correction_markers:
        pattern = re.compile(r'\b' + re.escape(marker) + r'\b', re.IGNORECASE)
        highlighted_text = pattern.sub(f'**{marker}**', highlighted_text)
    
    # Create a heatmap of token attention
    plt.figure(figsize=(20, 10))
    
    # Create a custom colormap from white to red
    cmap = LinearSegmentedColormap.from_list('WhiteToRed', ['white', 'red'])
    
    # Reshape attention for heatmap
    attention_matrix = attention.reshape(1, -1)
    
    # Plot heatmap
    sns.heatmap(attention_matrix, cmap=cmap, cbar=True, 
                xticklabels=tokens, yticklabels=['CLS'],
                annot=False)
    
    plt.title(f"{'Faithful' if is_faithful else 'Unfaithful'} Example: CLS Token Attention")
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{output_dir}/{model_name}_example_{example_id}_attention.png')
    plt.close()
    
    # Create a text file with the example and highlighted attention
    with open(f'{output_dir}/{model_name}_example_{example_id}_text.txt', 'w', encoding='utf-8') as f:
        f.write(f"Example ID: {example_id}\n")
        f.write(f"Classification: {'Faithful' if is_faithful else 'Unfaithful'}\n\n")
        f.write("Original Text:\n")
        f.write(text)
        f.write("\n\nHighlighted Text (correction markers in **bold**):\n")
        f.write(highlighted_text)
        f.write("\n\nTop 10 Tokens by Attention:\n")
        
        # Get top tokens
        token_attn_pairs = list(zip(tokens, attention))
        top_tokens = sorted(token_attn_pairs, key=lambda x: x[1], reverse=True)[:10]
        
        for i, (token, score) in enumerate(top_tokens):
            f.write(f"{i+1}. {token}: {score:.4f}\n")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = AttentionClassifier(args.model_path).to(device)
    
    # Load data
    data_dir = f'data/{args.model_name}'
    print(f"Loading data from {data_dir}")
    cots = load_yaml_files(data_dir)
    
    # Filter to only include examples with explicit faithfulness labels
    labeled_cots = [cot for cot in cots if 'is_faithful' in cot]
    
    print(f"Found {len(labeled_cots)} labeled examples")
    print(f"Faithful: {sum(1 for cot in labeled_cots if cot['is_faithful'])}")
    print(f"Unfaithful: {sum(1 for cot in labeled_cots if not cot['is_faithful'])}")
    
    # Create dataset and dataloader
    dataset = TokenDataset(labeled_cots, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Analyze attention weights
    results = analyze_attention_weights(model, dataloader, tokenizer, device, top_k=args.top_k, num_examples=args.num_examples)
    
    # Plot top tokens
    df = plot_top_tokens(results, args.output_dir, args.model_name.lower(), top_k=args.top_k)
    
    # Print top tokens
    print("\nTop tokens by attention in unfaithful classification:")
    for i, (token, ratio) in enumerate(df.iloc[:20].iterrows()):
        print(f"{i+1}. {token}: {ratio['Ratio']:.2f}x more attention in unfaithful examples")
    
    # Visualize examples
    print("\nVisualizing example attention patterns...")
    for i, example in enumerate(results['unfaithful_examples']):
        visualize_example_attention(example, tokenizer, args.output_dir, args.model_name.lower(), example['example_id'], False)
    
    for i, example in enumerate(results['faithful_examples']):
        visualize_example_attention(example, tokenizer, args.output_dir, args.model_name.lower(), example['example_id'], True)
    
    print(f"Analysis complete. Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 