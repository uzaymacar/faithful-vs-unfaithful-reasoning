import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import yaml
import glob
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformer_lens import HookedTransformer
from collections import defaultdict
from matplotlib import pyplot as plt

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Function to load and process YAML files
def load_yaml_files(data_dir):
    yaml_files = glob.glob(os.path.join(data_dir, "*.yaml"))
    print(f"Found {len(yaml_files)} YAML files in {data_dir}")
    
    all_cots = []
    
    for yaml_file in tqdm(yaml_files, desc="Loading YAML files"):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        for example_id, example_data in data.items():
            # Extract metadata
            metadata = example_data.get('metadata', {})
            prop_id = metadata.get('prop_id', '')
            question = metadata.get('q_str', '')
            prompt = example_data.get('prompt', '')
            correct_answer = metadata.get('answer', '')
            
            # Process faithful responses
            faithful_responses = example_data.get('faithful_responses', {})
            for response_id, response_data in faithful_responses.items():
                cot = {
                    'example_id': example_id,
                    'response_id': response_id,
                    'question': question,
                    'prop_id': prop_id,
                    'ground_truth_answer': correct_answer,
                    'response': response_data.get('response', ''),
                    'final_answer': response_data.get('final_answer', ''),
                    'result': response_data.get('result', ''),
                    'is_faithful': True,
                    'generated_cot': f"{prompt}{response_data.get('response', '')}"
                }
                all_cots.append(cot)
                
            # Process unfaithful responses (if any)
            unfaithful_responses = example_data.get('unfaithful_responses', {})
            for response_id, response_data in unfaithful_responses.items():
                cot = {
                    'example_id': example_id,
                    'response_id': response_id,
                    'question': question,
                    'prop_id': prop_id,
                    'ground_truth_answer': correct_answer,
                    'response': response_data.get('response', ''),
                    'final_answer': response_data.get('final_answer', ''),
                    'result': response_data.get('result', ''),
                    'is_faithful': False,
                    'generated_cot': f"{prompt}{response_data.get('response', '')}"
                }
                all_cots.append(cot)
    
    return all_cots

# Load the CoT data from YAML files
model_name = 'Llama-3.1-8B-Instruct'  # Change to 'Qwen2.5-1.5B-Instruct' for the other model
data_dir = f'data/{model_name}'
cots = load_yaml_files(data_dir)

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

# Analyze distribution by property ID and question with respect to faithfulness
def analyze_label_distribution(dataset, name):
    # Track counts by property ID
    prop_id_faithful = defaultdict(int)
    prop_id_unfaithful = defaultdict(int)
    prop_id_total = defaultdict(int)
    
    # Track counts by question
    question_faithful = defaultdict(int)
    question_unfaithful = defaultdict(int)
    question_total = defaultdict(int)
    
    # Track counts by example ID
    example_id_faithful = defaultdict(int)
    example_id_unfaithful = defaultdict(int)
    example_id_total = defaultdict(int)
    
    for cot in dataset:
        # Get example ID
        example_id = cot.get('example_id', 'unknown')
        
        # Get question
        question = cot.get('question', 'unknown')
        
        # Get property ID if available
        prop_id = cot.get('prop_id', 'unknown')
        
        # Update counts based on faithfulness
        if cot.get('is_faithful', False):
            prop_id_faithful[prop_id] += 1
            question_faithful[question] += 1
            example_id_faithful[example_id] += 1
        else:
            prop_id_unfaithful[prop_id] += 1
            question_unfaithful[question] += 1
            example_id_unfaithful[example_id] += 1
        
        # Update total counts
        prop_id_total[prop_id] += 1
        question_total[question] += 1
        example_id_total[example_id] += 1
    
    print(f"\n{name} Set Label Distribution:")
    print(f"Total examples: {len(dataset)}")
    print(f"Faithful examples: {sum(prop_id_faithful.values())}")
    print(f"Unfaithful examples: {sum(prop_id_unfaithful.values())}")
    
    # Print property ID distribution
    print("\nProperty ID Distribution by Faithfulness:")
    print("Property ID | Total | Faithful | Unfaithful | % Faithful")
    print("-" * 60)
    for prop_id in sorted(prop_id_total.keys()):
        total = prop_id_total[prop_id]
        faithful = prop_id_faithful[prop_id]
        unfaithful = prop_id_unfaithful[prop_id]
        percent_faithful = (faithful / total) * 100 if total > 0 else 0
        print(f"{prop_id:12} | {total:5} | {faithful:8} | {unfaithful:10} | {percent_faithful:8.2f}%")
    
    # Print top 10 questions with most biased distribution
    print("\nTop 10 Questions with Most Biased Distribution:")
    print("Question | Total | Faithful | Unfaithful | % Faithful")
    print("-" * 80)
    
    # Calculate bias as distance from 50%
    question_bias = {}
    for question, total in question_total.items():
        if total >= 5:  # Only consider questions with at least 5 examples
            faithful = question_faithful[question]
            percent_faithful = (faithful / total) * 100
            # Bias is distance from 50%
            bias = abs(percent_faithful - 50)
            question_bias[question] = bias
    
    # Print top biased questions
    for question in sorted(question_bias.keys(), key=lambda q: question_bias[q], reverse=True)[:10]:
        total = question_total[question]
        faithful = question_faithful[question]
        unfaithful = question_unfaithful[question]
        percent_faithful = (faithful / total) * 100
        
        # Truncate long questions
        q_display = question[:40] + "..." if len(question) > 40 else question
        print(f"{q_display:50} | {total:5} | {faithful:8} | {unfaithful:10} | {percent_faithful:8.2f}%")

# Analyze label distribution for each split
analyze_label_distribution(train_cots, "Train")
analyze_label_distribution(val_cots, "Validation")
analyze_label_distribution(test_cots, "Test")

# Load the appropriate model based on model_name
if model_name == 'Qwen2.5-1.5B-Instruct':
    model_path = 'Qwen/Qwen2.5-1.5B-Instruct'
elif model_name == 'Llama-3.1-8B-Instruct':
    model_path = 'meta-llama/Llama-3.1-8B-Instruct'
else:
    raise ValueError(f"Unsupported model: {model_name}")

print(f"Loading {model_path}...")
llm_model = HookedTransformer.from_pretrained(model_path)
llm_model = llm_model.to(device)
llm_model.eval()  # Set to evaluation mode

# Function to extract residual stream activations
def extract_residual_activations(cot, model, layer_indices=None, max_length=512):
    """Extract residual stream activations for a CoT example."""
    # If layer_indices is None, use all layers
    if layer_indices is None:
        layer_indices = list(range(model.cfg.n_layers))
    
    # Tokenize the CoT
    full_text = cot['generated_cot']
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
        layer_activations = cache[f'blocks.{layer_idx}.hook_resid_post']
        activations.append(layer_activations)
    
    # Stack activations along a new dimension (layer)
    stacked_activations = torch.stack(activations, dim=1)
    
    # Remove batch dimension (since we're processing one example at a time)
    stacked_activations = stacked_activations.squeeze(0)
    
    return stacked_activations, tokens.shape[1] - 1  # -1 to exclude BOS token

# Define which layers to use for classification
layer_indices = None
if model_name == 'Qwen2.5-1.5B-Instruct':
    layer_indices = [0, 5, 10, 15, 20, 25] 
elif model_name == 'Llama-3.1-8B-Instruct':
    # layer_indices = [6, 7, 8, 9, 10, 11] # Kinda worked well, first time (80% val acc but high val loss)
    # layer_indices = [12, 13, 14, 15, 16, 17] # This worked good, second time (83% val acc but still somewhat high val loss)
    # layer_indices = [18, 19, 20, 21, 22, 23]
    layer_indices = [12, 13, 15, 16, 17, 18]
else:
    raise ValueError(f"Unsupported model: {model_name}")
print(f"Using layers: {layer_indices}")

# Extract activations for all examples
print("Extracting activations...")
train_activations = []
for cot in tqdm(train_cots, desc="Extracting train activations"):
    try:
        activations, seq_length = extract_residual_activations(cot, llm_model, layer_indices)
        train_activations.append({
            'activations': activations,
            'label': 1 if cot['is_faithful'] else 0,
            'seq_length': seq_length
        })
    except Exception as e:
        print(f"Error processing example {cot['example_id']}: {e}")

val_activations = []
for cot in tqdm(val_cots, desc="Extracting validation activations"):
    try:
        activations, seq_length = extract_residual_activations(cot, llm_model, layer_indices)
        val_activations.append({
            'activations': activations,
            'label': 1 if cot['is_faithful'] else 0,
            'seq_length': seq_length
        })
    except Exception as e:
        print(f"Error processing example {cot['example_id']}: {e}")

test_activations = []
for cot in tqdm(test_cots, desc="Extracting test activations"):
    try:
        activations, seq_length = extract_residual_activations(cot, llm_model, layer_indices)
        test_activations.append({
            'activations': activations,
            'label': 1 if cot['is_faithful'] else 0,
            'seq_length': seq_length
        })
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

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Define the classifier model
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=768):
        super(Classifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Projection layer to map residual stream to BERT input dimension
        # Adjust the input dimension to account for multiple layers
        self.projection = nn.Linear(input_dim * len(layer_indices), hidden_dim)
        print(f"Classifier Projection Layer Shape: {self.projection.weight.shape}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # Binary classification
        )
    
    def forward(self, activations, attention_mask=None):
        # activations shape: [batch_size, num_layers, seq_len, hidden_dim]
        batch_size, num_layers, seq_len, hidden_dim = activations.shape
        
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
classifier = Classifier(hidden_dim).to(device)

# Training parameters
num_epochs = 30
learning_rate = 2e-5
weight_decay = 0.01

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Training loop
best_val_accuracy = 0.0
best_model = None

# Track metrics for plotting
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

print("Starting training...")
for epoch in range(num_epochs):
    # Training
    classifier.train()
    train_loss = 0.0
    train_preds = []
    train_labels = []
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        # Move batch to device
        activations = batch['activations'].to(device)
        labels = batch['label'].to(device)
        seq_lengths = batch['seq_length']
        
        # Create attention mask based on sequence lengths
        attention_mask = torch.zeros(activations.shape[0], activations.shape[2], device=device)
        for i, length in enumerate(seq_lengths):
            attention_mask[i, :length] = 1
        
        # Forward pass
        outputs = classifier(activations, attention_mask)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
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
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            # Move batch to device
            activations = batch['activations'].to(device)
            labels = batch['label'].to(device)
            seq_lengths = batch['seq_length']
            
            # Create attention mask based on sequence lengths
            attention_mask = torch.zeros(activations.shape[0], activations.shape[2], device=device)
            for i, length in enumerate(seq_lengths):
                attention_mask[i, :length] = 1
            
            # Forward pass
            outputs = classifier(activations, attention_mask)
            loss = criterion(outputs, labels)
            
            # Track metrics
            val_loss += loss.item()
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
plt.savefig(f'figures/classifier/{model_name.lower()}_training_curves.png')
plt.show()

print(f"Training curves saved to figures/classifier/{model_name.lower()}_training_curves.png")

# Load best model for evaluation
classifier.load_state_dict(best_model)
classifier.eval()

# Test evaluation
test_preds = []
test_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        # Move batch to device
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
os.makedirs('saved_models/classifier', exist_ok=True)
model_filename = f'saved_models/classifier/{model_name.lower()}_{"_".join(str(i) for i in layer_indices)}.pt'
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
    
    return {
        'layer_importance': layer_importance.cpu().numpy(),
        'position_importance': dict(position_importance)
    }

# Run feature importance analysis
print("\nAnalyzing feature importance...")
os.makedirs('figures/classifier', exist_ok=True)
importance_results = analyze_feature_importance(classifier, test_loader, device)

# Plot layer importance
plt.figure(figsize=(10, 6))
plt.bar(
    [f"Layer {layer_indices[i]}" for i in range(len(layer_indices))],
    importance_results['layer_importance']
)
plt.xlabel('Layer')
plt.ylabel('Importance Score')
plt.title('Layer Importance for Faithful versus Unfaithful Classification')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'figures/classifier/{model_name.lower()}_layer_importance.png')
plt.show()

# Plot position importance
plt.figure(figsize=(10, 6))
positions = sorted(importance_results['position_importance'].keys())
plt.bar(
    [f"Pos {pos}" for pos in positions],
    [importance_results['position_importance'][pos] for pos in positions]
)
plt.xlabel('Position')
plt.ylabel('Importance Score')
plt.title('Position Importance for Faithful versus Unfaithful Classification')
plt.tight_layout()
plt.savefig(f'figures/classifier/{model_name.lower()}_position_importance.png')
plt.show()

print(f"Analysis complete. Results saved to figures/classifier/{model_name.lower()}_layer_importance.png and figures/classifier/{model_name.lower()}_position_importance.png")