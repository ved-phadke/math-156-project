import argparse
import yaml
import torch
import torch.nn as nn
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import BaselineClassifier
from src.data_loader import get_filtered_mnist_dataloaders

def evaluate_task(config_path, eval_key):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    eval_config_entry = None
    for entry in config.get('evaluation', []):
        if entry['name'] == eval_key:
            eval_config_entry = entry
            break
    
    if not eval_config_entry:
        print(f"Evaluation entry {eval_key} not found in config.")
        return

    eval_config = eval_config_entry
    print(f"Starting evaluation for: {eval_config['name']}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    print(f"Loading data for digits: {eval_config['digits']}")
    _, test_loader = get_filtered_mnist_dataloaders(
        eval_config['digits'],
        batch_size_test=config.get('default_eval_batch_size', 256),
        data_root='./data' # Assumes script is run from project root
    )

    # Model
    model = BaselineClassifier().to(device)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    base_model_dir = os.path.join(project_root, 'models') # Standardized models directory
    load_path = os.path.join(base_model_dir, eval_config['model_load_name'])
    
    if os.path.exists(load_path):
        print(f"Loading model from {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=device))
    else:
        print(f"Error: Model path {load_path} not found. Cannot evaluate.")
        return

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0) # Accumulate loss correctly
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        print("No data found for evaluation. Check dataset and filters.")
        return 0.0

    avg_test_loss = test_loss / total
    accuracy = 100 * correct / total
    print(f'Evaluation: {eval_config["name"]}')
    print(f'  Test Loss: {avg_test_loss:.4f}')
    print(f'  Accuracy on {eval_config["task_name"]} ({eval_config["digits"]} digits): {accuracy:.2f}% ({correct}/{total})')
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the experiment YAML config file (e.g., experiments/config.yaml).")
    parser.add_argument('--eval_name', type=str, required=True, help="Name of the evaluation entry from the config file.")
    args = parser.parse_args()

    evaluate_task(args.config, args.eval_name)
