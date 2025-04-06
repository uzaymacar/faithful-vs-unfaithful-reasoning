import argparse
import os
import random
from collections import defaultdict
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformer_lens import HookedTransformer

from utils import load_yaml_files, analyze_label_distribution

def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier to distinguish faithful from unfaithful reasoning')
    parser.add_argument('-m', '--model_name', type=str, default='qwen2.5-1.5b-instruct', choices=['qwen2.5-1.5b-instruct', 'llama-3.1-8b-instruct'], help='Name of the model to analyze (default: qwen2.5-1.5b-instruct)')
    parser.add_argument('-a', '--activation_type', type=str, default='resid_post', choices=['resid_post', 'tokens'], help='Type of activations to use for classification (default: resid_post)')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size for training (default: 8)')
    parser.add_argument('-e', '--num_epochs', type=int, default=30, help='Number of training epochs (default: 30)')
    parser.add_argument('-l', '--learning_rate', type=float, default=2e-5, help='Learning rate (default: 2e-5)')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('-o', '--output_dir', type=str, default='figures', help='Directory to save output figures (default: figures)')
    parser.add_argument('-d', '--model_dir', type=str, default='saved_models', help='Directory to save trained models (default: saved_models)')
    parser.add_argument('-n', '--num_examples', type=int, default=None, help='Number of examples to use for training (default: None)')
    return parser.parse_args()

args = parse_args()

# Set random seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Use the model name from arguments
model_name = args.model_name
data_dir = f'data/{model_name}'
print(f"Using model: {model_name}")
print(f"Loading data from: {data_dir}")

# Create output directories with activation type
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.model_dir, exist_ok=True)
os.makedirs(f"{args.output_dir}/classifier_{args.activation_type}", exist_ok=True)
os.makedirs(f"{args.model_dir}/classifier_{args.activation_type}", exist_ok=True)

# Load the CoT data from YAML files
cots = load_yaml_files(data_dir)[0:args.num_examples]

# Filter to only include examples with explicit faithfulness labels
labeled_cots = [cot for cot in cots]
faithful_cots = [cot for cot in labeled_cots if cot['is_faithful'] == True]
unfaithful_cots = [cot for cot in labeled_cots if cot['is_faithful'] == False]
print(f"Total labeled CoTs: {len(labeled_cots)}")
print(f"Faithful CoTs: {len(faithful_cots)}")
print(f"Unfaithful CoTs: {len(unfaithful_cots)}")

# Create balanced dataset by downsampling the majority class
min_class_size = min(len(faithful_cots), len(unfaithful_cots))
if len(faithful_cots) > min_class_size:
    faithful_cots = random.sample(faithful_cots, min_class_size)
if len(unfaithful_cots) > min_class_size:
    unfaithful_cots = random.sample(unfaithful_cots, min_class_size)

# Combine and shuffle
balanced_cots = faithful_cots + unfaithful_cots
random.shuffle(balanced_cots)

print(f"Balanced dataset size: {len(balanced_cots)}")
print(f"Faithful: {len(faithful_cots)}, Unfaithful: {len(unfaithful_cots)}")

# Split into train, validation, and test sets (70/15/15)
train_size = int(0.7 * len(balanced_cots))
val_size = int(0.15 * len(balanced_cots))
test_size = len(balanced_cots) - train_size - val_size

train_cots = balanced_cots[:train_size]
val_cots = balanced_cots[train_size:train_size+val_size]
test_cots = balanced_cots[train_size+val_size:]

print(f"Train set: {len(train_cots)}")
print(f"Validation set: {len(val_cots)}")
print(f"Test set: {len(test_cots)}")

# Check balance in each split
print(f"Train set balance: {sum(1 for cot in train_cots if cot['is_faithful'])}/{len(train_cots)}")
print(f"Validation set balance: {sum(1 for cot in val_cots if cot['is_faithful'])}/{len(val_cots)}")
print(f"Test set balance: {sum(1 for cot in test_cots if cot['is_faithful'])}/{len(test_cots)}")

# Check balance in each split
print(f"Train set balance: {sum(1 for cot in train_cots if cot['is_faithful'])}/{len(train_cots)}")
print(f"Validation set balance: {sum(1 for cot in val_cots if cot['is_faithful'])}/{len(val_cots)}")
print(f"Test set balance: {sum(1 for cot in test_cots if cot['is_faithful'])}/{len(test_cots)}")

# Analyze label distribution for each split
analyze_label_distribution(train_cots, "Train")
analyze_label_distribution(val_cots, "Validation")
analyze_label_distribution(test_cots, "Test")

# Load the appropriate model based on model_name
if model_name == 'qwen2.5-1.5b-instruct':
    model_path = 'Qwen/Qwen2.5-1.5B-Instruct'
elif model_name == 'llama-3.1-8b-instruct':
    model_path = 'meta-llama/Llama-3.1-8B-Instruct'
else:
    raise ValueError(f"Unsupported model: {model_name}")

print(f"Loading {model_path}...")
llm_model = HookedTransformer.from_pretrained(model_path)
llm_model = llm_model.to(device)
llm_model.eval()  # Set to evaluation mode

# Function to extract activations
def extract_activations(cot, model, layer_indices=None, max_length=512):
    """Extract activations for a CoT example based on specified activation type."""
    # If layer_indices is None, use all layers
    if layer_indices is None:
        layer_indices = list(range(model.cfg.n_layers))
    
    # Tokenize the CoT
    full_text = cot['generated_cot']
        
    if args.activation_type == 'tokens':
        # For token-based classification, use BERT tokenizer directly
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Tokenize with BERT tokenizer
        encoded = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors='pt')
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        return input_ids, attention_mask, input_ids.shape[1]
    else:
        tokens = model.to_tokens(full_text, prepend_bos=True)
        
        # Truncate if too long
        if tokens.shape[1] > max_length:
            tokens = tokens[:, :max_length]
        
        # Run the model and cache activations
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        
        # Extract residual stream activations for specified layers
        activations = []
        for layer_idx in layer_indices:
            layer_activations = cache[f'blocks.{layer_idx}.hook_{args.activation_type}']
            activations.append(layer_activations)
        
        # Stack activations along a new dimension (layer)
        stacked_activations = torch.stack(activations, dim=1)
        
        # Remove batch dimension (since we're processing one example at a time)
        stacked_activations = stacked_activations.squeeze(0)
        
        return stacked_activations, tokens.shape[1] - 1  # -1 to exclude BOS token

# Define which layers to use for classification
# NOTE: Layer indices were found empirically after training the model for a few iterations
layer_indices = None
if model_name == 'qwen2.5-1.5b-instruct':
    layer_indices = [0, 5, 10, 15, 20, 25] 
elif model_name == 'llama-3.1-8b-instruct':
    layer_indices = [12, 13, 14, 15, 16, 17]
else:
    raise ValueError(f"Unsupported model: {model_name}")
print(f"Using layers: {layer_indices}")

# Extract activations for all examples
print("Extracting activations...")
train_activations = []
for cot in tqdm(train_cots, desc="Extracting train activations"):
    try:
        if args.activation_type == 'tokens':
            input_ids, attention_mask, seq_length = extract_activations(cot, llm_model, layer_indices)
            train_activations.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'label': 1 if cot['is_faithful'] else 0, 'seq_length': seq_length})
        else:
            activations, seq_length = extract_activations(cot, llm_model, layer_indices)
            train_activations.append({'activations': activations, 'label': 1 if cot['is_faithful'] else 0, 'seq_length': seq_length})
    except Exception as e:
        print(f"Error processing example {cot['example_id']}: {e}")

val_activations = []
for cot in tqdm(val_cots, desc="Extracting validation activations"):
    try:
        if args.activation_type == 'tokens':
            input_ids, attention_mask, seq_length = extract_activations(cot, llm_model, layer_indices)
            val_activations.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'label': 1 if cot['is_faithful'] else 0, 'seq_length': seq_length})
        else:
            activations, seq_length = extract_activations(cot, llm_model, layer_indices)
            val_activations.append({'activations': activations, 'label': 1 if cot['is_faithful'] else 0, 'seq_length': seq_length})
    except Exception as e:
        print(f"Error processing example {cot['example_id']}: {e}")

test_activations = []
for cot in tqdm(test_cots, desc="Extracting test activations"):
    try:
        if args.activation_type == 'tokens':
            input_ids, attention_mask, seq_length = extract_activations(cot, llm_model, layer_indices)
            test_activations.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'label': 1 if cot['is_faithful'] else 0, 'seq_length': seq_length})
        else:
            activations, seq_length = extract_activations(cot, llm_model, layer_indices)
            test_activations.append({'activations': activations, 'label': 1 if cot['is_faithful'] else 0, 'seq_length': seq_length })

    except Exception as e:
        print(f"Error processing example {cot['example_id']}: {e}")

print(f"Extracted activations for {len(train_activations)} train, {len(val_activations)} validation, and {len(test_activations)} test examples")

# Dataset class for residual stream activations
class Dataset(Dataset):
    def __init__(self, activations_data):
        self.activations_data = activations_data
    
    def __len__(self):
        return len(self.activations_data)
    
    def __getitem__(self, idx):
        return self.activations_data[idx]

# Collate function for variable-length sequences
def collate_fn(batch):
    # Check if batch is empty
    if len(batch) == 0:
        return {'activations': torch.tensor([]), 'label': torch.tensor([]), 'seq_length': []}
    
    if args.activation_type == 'tokens':
        # For token-based approach
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['label'] for item in batch]
        
        # Pad sequences
        max_len = max(ids.shape[1] for ids in input_ids)
        
        padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        padded_masks = torch.zeros(len(batch), max_len, dtype=torch.long)
        label_tensor = torch.zeros(len(batch), dtype=torch.long)
        
        for i, (ids, mask, label) in enumerate(zip(input_ids, attention_masks, labels)):
            seq_len = ids.shape[1]
            padded_ids[i, :seq_len] = ids[0, :seq_len]
            padded_masks[i, :seq_len] = mask[0, :seq_len]
            label_tensor[i] = label
        
        return {
            'input_ids': padded_ids,
            'attention_mask': padded_masks,
            'label': label_tensor
        }
    else:
        # Original collate function for residual activations
        # Find max sequence length in the batch
        max_seq_len = max(item['activations'].shape[1] for item in batch)
        
        # Get other dimensions
        batch_size = len(batch)
        num_layers = batch[0]['activations'].shape[0]
        hidden_dim = batch[0]['activations'].shape[2]
        
        # Initialize tensors
        activations = torch.zeros(batch_size, num_layers, max_seq_len, hidden_dim)
        labels = torch.zeros(batch_size, dtype=torch.long)
        seq_lengths = []
        
        # Fill tensors
        for i, item in enumerate(batch):
            seq_len = item['activations'].shape[1]
            activations[i, :, :seq_len, :] = item['activations'][:, :seq_len, :]
            labels[i] = item['label']
            seq_lengths.append(seq_len)
        
        return {
            'activations': activations,
            'label': labels,
            'seq_length': seq_lengths
        }

# Create datasets and dataloaders
train_dataset = Dataset(train_activations)
val_dataset = Dataset(val_activations)
test_dataset = Dataset(test_activations)

batch_size = args.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Define the classifier model
class Classifier(nn.Module):
    def __init__(self, input_dim, projection_dim=768):
        super(Classifier, self).__init__()
        self.projection_dim = projection_dim
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        if args.activation_type != 'tokens':            
            self.projection = nn.Linear(input_dim * len(layer_indices), projection_dim)
            print(f"Classifier projection layer shape: {self.projection.weight.shape}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, 2)  # Binary classification
        )
    
    def forward(self, x, attention_mask=None):
        if args.activation_type == 'tokens':
            # For token-based classification, x is input_ids
            outputs = self.bert(input_ids=x, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
        else: 
            # Original forward pass for residual stream activations
            activations = x  # Rename for clarity
            batch_size, num_layers, seq_len, hidden_dim = activations.shape
            # For Qwen: Exactly the same as original
            # Reshape to [batch_size * seq_len, num_layers * hidden_dim]
            activations = activations.permute(0, 2, 1, 3).reshape(batch_size * seq_len, num_layers * hidden_dim)
            
            # Project to BERT input dimension
            projected = self.projection(activations)
            
            # Reshape back to [batch_size, seq_len, hidden_dim]
            projected = projected.reshape(batch_size, seq_len, -1)
            
            # Create attention mask if not provided
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, seq_len, device=activations.device)
            
            # Pass through BERT
            outputs = self.bert(inputs_embeds=projected, attention_mask=attention_mask)
            
            # Use the [CLS] token representation for classification
            cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through classification head
        logits = self.classifier(cls_output)
        
        return logits

# Initialize the classifier
hidden_dim = llm_model.cfg.d_model
projection_dim = 768
print(f"Projection dim: {projection_dim}")
classifier = Classifier(input_dim=hidden_dim, projection_dim=projection_dim).to(device)

# Training parameters
num_epochs = args.num_epochs
learning_rate = args.learning_rate
weight_decay = 0.01

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

# Training loop
best_val_accuracy = 0.0
best_model = None

# Track metrics for plotting
train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

print("Starting training...")
for epoch in range(num_epochs):
    # Training
    classifier.train()
    train_loss = 0.0
    train_preds, train_labels = [], []
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        # Move batch to device
        if args.activation_type == 'tokens':
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = classifier(input_ids, attention_mask)
        else:
            activations = batch['activations'].to(device)
            labels = batch['label'].to(device)
            seq_lengths = batch['seq_length']
            
            # Create attention mask based on sequence lengths
            attention_mask = torch.zeros(activations.shape[0], activations.shape[2], device=device)
            for i, length in enumerate(seq_lengths):
                attention_mask[i, :length] = 1
            
            # Forward pass
            outputs = classifier(activations, attention_mask)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
    
    # Calculate training metrics
    epoch_train_loss = train_loss / len(train_loader)
    train_accuracy = accuracy_score(train_labels, train_preds)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(train_labels, train_preds, average='binary')
    
    # Store for plotting
    train_losses.append(epoch_train_loss)
    train_accuracies.append(train_accuracy)
    
    # Validation
    classifier.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            # Move batch to device
            if args.activation_type == 'tokens':
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = classifier(input_ids, attention_mask)
            else:
                activations = batch['activations'].to(device)
                labels = batch['label'].to(device)
                seq_lengths = batch['seq_length']
                
                # Create attention mask based on sequence lengths
                attention_mask = torch.zeros(activations.shape[0], activations.shape[2], device=device)
                for i, length in enumerate(seq_lengths):
                    attention_mask[i, :length] = 1
                
                # Forward pass
                outputs = classifier(activations, attention_mask)
            
            # Track metrics
            val_loss += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    # Calculate validation metrics
    epoch_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')
    
    # Store for plotting
    val_losses.append(epoch_val_loss)
    val_accuracies.append(val_accuracy)
    
    # Print epoch results
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {epoch_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
    print(f"  Val Loss: {epoch_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
    
    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model = classifier.state_dict().copy()
        print(f"  New best model saved with validation accuracy: {val_accuracy:.4f}")

# Plot training and validation curves
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(f'{args.output_dir}/classifier_{args.activation_type}/{model_name.lower()}_training_curves.png')
plt.show()

print(f"Training curves saved to {args.output_dir}/classifier_{args.activation_type}/{model_name.lower()}_training_curves.png")

# Load best model for evaluation
classifier.load_state_dict(best_model)
classifier.eval()

# Test evaluation
test_preds = []
test_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        # Move batch to device
        if args.activation_type == 'tokens':
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = classifier(input_ids, attention_mask)
        else:
            activations = batch['activations'].to(device)
            labels = batch['label'].to(device)
            seq_lengths = batch['seq_length']
            
            # Create attention mask based on sequence lengths
            attention_mask = torch.zeros(activations.shape[0], activations.shape[2], device=device)
            for i, length in enumerate(seq_lengths):
                attention_mask[i, :length] = 1
            
            # Forward pass
            outputs = classifier(activations, attention_mask)
        
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# Calculate test metrics
test_accuracy = accuracy_score(test_labels, test_preds)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary')

print("\nTest Results:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")

# Save the model with model name in the filename
os.makedirs(f'{args.model_dir}/classifier_{args.activation_type}', exist_ok=True)
model_filename = f'{args.model_dir}/classifier_{args.activation_type}/{model_name.lower()}_{"_".join(str(i) for i in layer_indices)}.pt'
torch.save(classifier.state_dict(), model_filename)
print(f"Model saved to {model_filename}")

# Feature importance analysis
def analyze_feature_importance(model, dataloader, device):
    """Analyze which layers/positions contribute most to classification decisions."""
    model.eval()
    layer_importance = torch.zeros(len(layer_indices), device=device)
    position_importance = defaultdict(float)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing features"):
            activations = batch['activations'].to(device)
            labels = batch['label'].to(device)
            seq_lengths = batch['seq_length']
            
            batch_size, num_layers, seq_len, hidden_dim = activations.shape
            
            # Create attention mask based on sequence lengths
            attention_mask = torch.zeros(batch_size, seq_len, device=device)
            for i, length in enumerate(seq_lengths):
                attention_mask[i, :length] = 1
            
            # Baseline prediction
            outputs = model(activations, attention_mask)
            baseline_probs = torch.softmax(outputs, dim=1)
            
            # Analyze layer importance
            for layer_idx in range(num_layers):
                # Zero out this layer
                zeroed_activations = activations.clone()
                zeroed_activations[:, layer_idx, :, :] = 0
                
                # Get new prediction
                zeroed_outputs = model(zeroed_activations, attention_mask)
                zeroed_probs = torch.softmax(zeroed_outputs, dim=1)
                
                # Calculate impact on correct class probability
                correct_class_impact = torch.abs(
                    baseline_probs[torch.arange(batch_size), labels] - 
                    zeroed_probs[torch.arange(batch_size), labels]
                ).mean()
                
                layer_importance[layer_idx] += correct_class_impact.item()
            
            # Analyze position importance
            for pos in range(seq_len):
                # Zero out this position across all layers
                zeroed_activations = activations.clone()
                zeroed_activations[:, :, pos, :] = 0
                
                # Get new prediction
                zeroed_outputs = model(zeroed_activations, attention_mask)
                zeroed_probs = torch.softmax(zeroed_outputs, dim=1)
                
                # Calculate impact on correct class probability
                correct_class_impact = torch.abs(
                    baseline_probs[torch.arange(batch_size), labels] - 
                    zeroed_probs[torch.arange(batch_size), labels]
                ).mean()
                
                position_importance[pos] += correct_class_impact.item()
    
    # Normalize by number of batches
    num_batches = len(dataloader)
    layer_importance /= num_batches
    for pos in position_importance:
        position_importance[pos] /= num_batches
    
    return { 'layer_importance': layer_importance.cpu().numpy(), 'position_importance': dict(position_importance) }

# Run feature importance analysis
if args.activation_type != 'tokens':
    print("\nAnalyzing feature importance...")
    os.makedirs(f'{args.output_dir}/classifier_{args.activation_type}', exist_ok=True)
    importance_results = analyze_feature_importance(classifier, test_loader, device)

    # Plot layer importance
    plt.figure(figsize=(10, 6))
    plt.bar([f"Layer {layer_indices[i]}" for i in range(len(layer_indices))], importance_results['layer_importance'])
    plt.xlabel('Layer')
    plt.ylabel('Importance Score')
    plt.title('Layer Importance for Faithful versus Unfaithful Classification')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/classifier_{args.activation_type}/{model_name.lower()}_layer_importance.png')
    plt.show()

    # Plot position importance
    plt.figure(figsize=(10, 6))
    positions = sorted(importance_results['position_importance'].keys())
    plt.bar([f"Pos {pos}" for pos in positions], [importance_results['position_importance'][pos] for pos in positions])
    plt.xlabel('Position')
    plt.ylabel('Importance Score')
    plt.title('Position Importance for Faithful versus Unfaithful Classification')
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/classifier_{args.activation_type}/{model_name.lower()}_position_importance.png')
    plt.show()

    print(f"Analysis complete. Results saved to {args.output_dir}/classifier_{args.activation_type}/{model_name.lower()}_layer_importance.png and {args.output_dir}/classifier_{args.activation_type}/{model_name.lower()}_position_importance.png")