import argparse
import yaml
import copy
from collections import defaultdict
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import BaselineClassifier
from src.data_loader import get_filtered_mnist_dataloaders

def train_task(config_path, task_key):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if task_key not in config:
        print(f"Task {task_key} not found in config.")
        return

    task_config = config[task_key]
    print(f"Starting training for task: {task_config['name']}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print(f"Loading data for digits: {task_config['digits']}")
    train_loader, _ = get_filtered_mnist_dataloaders(
        task_config['digits'],
        batch_size_train=task_config['train_params'].get('batch_size', 128),
        data_root='./data'
    )

    # Model
    model = BaselineClassifier().to(device)
    
    # Determine absolute path for base_model_path relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    base_model_dir = os.path.join(project_root, 'models')

    if task_config.get('model_load_name'):
        load_path = os.path.join(base_model_dir, task_config['model_load_name'])
        if os.path.exists(load_path):
            print(f"Loading model from {load_path}")
            model.load_state_dict(torch.load(load_path, map_location=device))
        else:
            print(f"Warning: Model path {load_path} not found. Starting with a fresh model.")

    # Optimizer and Loss
    optimizer_name = task_config['train_params'].get('optimizer', 'RMSprop').lower()
    lr = task_config['train_params'].get('lr', 0.001)
    if optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        print(f"Unsupported optimizer: {optimizer_name}. Defaulting to SGD.")
        optimizer = optim.SGD(model.parameters(), lr=lr)

    criterion = None

    use_ewc = task_config['train_params'].get('ewc', False)

    if use_ewc:
        ewc_lambda = task_config['train_params'].get('ewc_lambda', 1)

        params_A = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

        def compute_fisher(model, data_loader, num_samples=2000):
            fisher = defaultdict(float)
            model.eval()
            
            total_seen = 0
            
            for inputs, targets in data_loader:
                batch_size = inputs.size(0)
                if total_seen >= num_samples:
                    break
                
                if total_seen + batch_size > num_samples:
                    batch_size = num_samples - total_seen
                    inputs, targets = inputs[:batch_size], targets[:batch_size]
        
                inputs, targets = inputs.to(device), targets.to(device)
                model.zero_grad()
                outputs = model(inputs)
                log_probs = F.log_softmax(outputs, dim=1)
                loss = F.nll_loss(log_probs, targets)
                loss.backward()
        
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        fisher[name] += (param.grad.detach() ** 2) * batch_size
                
                total_seen += batch_size
        
            for name in fisher:
                fisher[name] /= total_seen
        
            return fisher

        task_a_loader, _ = get_filtered_mnist_dataloaders(
            config['task1']['digits'],
            batch_size_train=config['task1']['train_params'].get('batch_size', 128),
            data_root='./data'
        )

        print(f"Computing Fisher coefficients for model {task_config.get('model_load_name')}")
        fisher_A = compute_fisher(model, task_a_loader)

        def ewc_loss(model, loss_taskB):
            ewc_reg = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ewc_reg += (fisher_A[name] * (param - params_A[name]) ** 2).sum()
            return loss_taskB + (ewc_lambda / 2) * ewc_reg

    if task_config['train_params'].get('use_masked_training', False):
        def masked_cross_entropy(logits, targets):
            """
            logits: [batch_size, 10]
            targets: [batch_size] with values in valid classes
            """
            # Mask irrelevant logits
            mask = torch.zeros_like(logits)
            mask[:, task_config['digits']] = 1
            masked_logits = logits * mask
        
            # Re-normalize masked logits
            masked_probs = F.log_softmax(masked_logits, dim=1)
        
            return F.nll_loss(masked_probs, targets)

        criterion = masked_cross_entropy
    else:
        criterion = nn.CrossEntropyLoss()

    # Training Loop
    model.train()
    epochs = task_config['train_params'].get('epochs', 10)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if use_ewc:
                loss = ewc_loss(model, loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0: # Log every 100 batches
                print(f'Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
        print(f'Epoch [{epoch + 1}/{epochs}] completed.')

    # Save model
    if not os.path.exists(base_model_dir):
        os.makedirs(base_model_dir)
    
    save_path = os.path.join(base_model_dir, task_config['model_save_name'])
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on a specific task.")
    parser.add_argument('--config', type=str, required=True, help="Path to the experiment YAML config file (e.g., experiments/config.yaml).")
    parser.add_argument('--task', type=str, required=True, help="Task key from the config file (e.g., task1, task2).")
    args = parser.parse_args()
    
    train_task(args.config, args.task)
