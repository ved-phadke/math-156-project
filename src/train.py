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

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import BaselineClassifier
from src.data_loader import get_filtered_mnist_dataloaders

def train_task(config_path, task_key_to_train, learning_paradigm_arg):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if task_key_to_train not in config:
        print(f"Task {task_key_to_train} not found in config.")
        return

    task_config = config[task_key_to_train]
    experiment_paradigm = config.get('learning_paradigm', learning_paradigm_arg) # Prefer config, fallback to arg

    print(f"Starting training for task: {task_config['name']} with paradigm: {experiment_paradigm}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine num_classes for the model
    if experiment_paradigm == "TIL":
        num_classes_current_task = len(list(set(task_config['digits'])))
    elif experiment_paradigm == "CIL":
        num_classes_current_task = 10 # Assuming MNIST 0-9 for CIL
    else:
        raise ValueError(f"Unsupported learning_paradigm: {experiment_paradigm}")

    # Data Loader
    print(f"Loading data for digits: {task_config['digits']}")
    # Determine if labels should be remapped based on the paradigm
    should_remap_labels = True if experiment_paradigm == "TIL" else False
    print(f"Remapping labels: {should_remap_labels} (Paradigm: {experiment_paradigm})")

    train_loader, _ = get_filtered_mnist_dataloaders(
        digits=task_config['digits'],
        batch_size_train=task_config['train_params'].get('batch_size', 128),
        data_root=config.get('data_dir', './data'), # Use data_dir from config
        remap_labels=should_remap_labels # Pass the boolean flag
    )

    # Model Initialization
    model = BaselineClassifier(num_classes=num_classes_current_task).to(device)
    
    base_model_dir = os.path.join(project_root, config.get('base_model_dir', 'models'))
    os.makedirs(base_model_dir, exist_ok=True)

    if task_config.get('model_load_name'):
        load_path = os.path.join(base_model_dir, task_config['model_load_name'])
        if os.path.exists(load_path):
            print(f"Loading model weights from {load_path}")
            if experiment_paradigm == "TIL":
                # For TIL, load previous model, copy body, reinitialize head (fc2)
                # Determine num_classes of the model being loaded
                # This requires knowing which task it was saved from, or storing num_classes with model.
                # For simplicity, we assume model_load_name was trained with its own task-specific head.
                # We load its state_dict and copy compatible layers.
                
                loaded_state_dict = torch.load(load_path, map_location=device)
                current_model_dict = model.state_dict()
                
                # Filter out fc2 layer from loaded_state_dict if shapes mismatch or simply always for TIL new head
                # Here, we copy all layers that match in name and shape, except fc2 which is new.
                # fc2 of the current model will be randomly initialized.
                print("TIL: Transferring weights for layers other than fc2.")
                for name, param in loaded_state_dict.items():
                    if 'fc2' not in name and name in current_model_dict:
                        if param.shape == current_model_dict[name].shape:
                            current_model_dict[name].copy_(param)
                        else:
                            print(f"Shape mismatch for {name}: loaded {param.shape}, current {current_model_dict[name].shape}. Skipping.")
                    elif 'fc2' in name:
                         print(f"Skipping fc2 layer ('{name}') from loaded model for TIL new head.")
                model.load_state_dict(current_model_dict) # Load the modified dict
            else: # CIL
                model.load_state_dict(torch.load(load_path, map_location=device))
        else:
            print(f"Warning: Model path {load_path} not found. Starting with a fresh model.")
    
    # Optimizer (re-initialize if TIL and loaded a model, to catch new fc2 params)
    optimizer_name = task_config['train_params'].get('optimizer', 'RMSprop').lower()
    lr = task_config['train_params'].get('lr', 0.001)
    if optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=task_config['train_params'].get('alpha', 0.9))
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
        if experiment_paradigm == "CIL":
            def masked_cross_entropy(logits, targets):
                mask = torch.zeros_like(logits)
                mask[:, task_config['digits']] = 1
                masked_logits = logits * mask
                masked_probs = F.log_softmax(masked_logits, dim=1)
                return F.nll_loss(masked_probs, targets)
            criterion = masked_cross_entropy
        else:
            print("Masked training is only needed for CIL. Using CrossEntropyLoss for TIL.")
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Training Loop
    model.train()
    epochs = task_config['train_params'].get('epochs', 10)
    print(f"Training for {epochs} epochs...")
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), task_config['train_params'].get('max_grad_norm', 1.0))
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
        print(f'Epoch [{epoch + 1}/{epochs}] completed.')

    # Save model
    save_path = os.path.join(base_model_dir, task_config['model_save_name'])
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on a specific task.")
    parser.add_argument('--config', type=str, required=True, help="Path to the experiment YAML config file.")
    parser.add_argument('--task', type=str, required=True, help="Task key from the config file (e.g., task1, task2).")
    parser.add_argument('--learning_paradigm', type=str, default="CIL", choices=["CIL", "TIL"], help="Learning paradigm: CIL or TIL.")
    args = parser.parse_args()
    
    # Config path relative to project root if not absolute
    abs_config_path = args.config
    if not os.path.isabs(abs_config_path):
        abs_config_path = os.path.join(project_root, args.config)
        
    train_task(abs_config_path, args.task, args.learning_paradigm)
