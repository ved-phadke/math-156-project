import argparse
import yaml
import torch
import torch.nn as nn
import os
import sys
import csv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import BaselineClassifier
from src.data_loader import get_filtered_mnist_dataloaders

def evaluate_task(config_path, eval_name_key, learning_paradigm_arg):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    eval_entry_config = None
    for entry in config.get('evaluation', []):
        if entry['name'] == eval_name_key:
            eval_entry_config = entry
            break
    
    if not eval_entry_config:
        print(f"Evaluation entry {eval_name_key} not found in config.")
        return

    experiment_paradigm = eval_entry_config.get('learning_paradigm', config.get('learning_paradigm', learning_paradigm_arg))
    print(f"Starting evaluation for: {eval_entry_config['name']} with paradigm: {experiment_paradigm}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine num_classes for model and data
    # For data loader:
    eval_digits = config[eval_entry_config['eval_task_key']]['digits']
    
    # For model:
    base_model_dir = os.path.join(project_root, config.get('base_model_dir', 'models'))

    if experiment_paradigm == "TIL":
        if 'model_body_load_name' in eval_entry_config and 'model_head_load_name' in eval_entry_config:
            # TIL Head Swapping Logic
            print("TIL Head Swapping Evaluation")
            num_classes_for_head = len(list(set(config[eval_entry_config['head_task_key']]['digits'])))
            model = BaselineClassifier(num_classes=num_classes_for_head).to(device)
            
            # Load body
            body_load_path = os.path.join(base_model_dir, eval_entry_config['model_body_load_name'])
            if os.path.exists(body_load_path):
                print(f"Loading body from: {body_load_path}")
                body_state_dict = torch.load(body_load_path, map_location=device)
                model_dict = model.state_dict()
                # Copy all except fc2 from body_state_dict
                for name, param in body_state_dict.items():
                    if 'fc2' not in name and name in model_dict: # model_dict has the target fc2 shape
                        if param.shape == model_dict[name].shape:
                             model_dict[name].copy_(param)
                model.load_state_dict(model_dict) # Load partial
            else:
                print(f"Error: Body model path {body_load_path} not found.")
                return

            # Load head
            head_load_path = os.path.join(base_model_dir, eval_entry_config['model_head_load_name'])
            if os.path.exists(head_load_path):
                print(f"Loading head (fc2) from: {head_load_path}")
                head_state_dict = torch.load(head_load_path, map_location=device)
                # Copy only fc2 from head_state_dict
                if 'fc2.weight' in head_state_dict and 'fc2.bias' in head_state_dict:
                    if model.fc2.weight.shape == head_state_dict['fc2.weight'].shape:
                        model.fc2.weight.data.copy_(head_state_dict['fc2.weight'])
                        model.fc2.bias.data.copy_(head_state_dict['fc2.bias'])
                    else:
                        print(f"FC2 shape mismatch: model needs {model.fc2.weight.shape}, loaded head has {head_state_dict['fc2.weight'].shape}")
                        return
                else:
                    print("fc2 weights not found in head model state_dict.")
                    return
            else:
                print(f"Error: Head model path {head_load_path} not found.")
                return
            num_classes_for_data_loader = num_classes_for_head # Data remapped to head's classes
        else: # Standard TIL evaluation (model trained on task, evaluated on same task)
            num_classes_for_model = len(list(set(config[eval_entry_config['eval_task_key']]['digits'])))
            model = BaselineClassifier(num_classes=num_classes_for_model).to(device)
            load_path = os.path.join(base_model_dir, eval_entry_config['model_load_name'])
            if os.path.exists(load_path):
                print(f"Loading model from {load_path}")
                model.load_state_dict(torch.load(load_path, map_location=device))
            else:
                print(f"Error: Model path {load_path} not found.")
                return
            num_classes_for_data_loader = num_classes_for_model
            
    elif experiment_paradigm == "CIL":
        model = BaselineClassifier(num_classes=10).to(device) # CIL MNIST always 10 classes
        load_path = os.path.join(base_model_dir, eval_entry_config['model_load_name'])
        if os.path.exists(load_path):
            print(f"Loading model from {load_path}")
            model.load_state_dict(torch.load(load_path, map_location=device))
        else:
            print(f"Error: Model path {load_path} not found.")
            return
    else:
        raise ValueError(f"Unsupported learning_paradigm: {experiment_paradigm}")

    # Data Loader
    print(f"Loading evaluation data for digits: {eval_digits}")
    # Determine if labels should be remapped based on the paradigm
    should_remap_labels_eval = True if experiment_paradigm == "TIL" else False
    print(f"Remapping labels for evaluation: {should_remap_labels_eval} (Paradigm: {experiment_paradigm})")

    _, test_loader = get_filtered_mnist_dataloaders(
        digits=eval_digits,
        batch_size_test=eval_entry_config.get('batch_size_test', 256),
        data_root=config.get('data_dir', './data'),
        remap_labels=should_remap_labels_eval # Pass the boolean flag
    )
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss() # For calculating loss if needed, not strictly for accuracy
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Bias correction for CIL
            if experiment_paradigm == "CIL" and 'bias_correction' in eval_entry_config:
                bias  = eval_entry_config.get('bias_correction', 0.0)
                task1_digits = config['task1']['digits']
                for digit in task1_digits:
                    outputs[:, digit] += bias

            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        print("No data found for evaluation. Check dataset and filters.")
        avg_test_loss = 0
        accuracy = 0
    else:
        avg_test_loss = test_loss / total
        accuracy = 100 * correct / total
    
    print(f'Evaluation Complete: {eval_entry_config["name"]}')
    print(f'  Digits Evaluated: {eval_digits}')
    print(f'  Test Loss: {avg_test_loss:.4f}')
    print(f'  Accuracy: {accuracy:.2f}% ({correct}/{total})')
    
    # Save results
    results_dir = os.path.join(project_root, config.get('results_dir', 'results'))
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, config.get('results_filename', 'results.csv'))
    
    file_exists = os.path.isfile(results_file)
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists: # Write header if new file
            writer.writerow(["EvaluationName", "Paradigm", "EvaluatedDigits", "TestLoss", "Accuracy", "Correct", "Total", "ModelLoaded"])
        
        model_info = eval_entry_config.get('model_load_name', 
                       f"Body:{eval_entry_config.get('model_body_load_name')}_Head:{eval_entry_config.get('model_head_load_name')}")

        writer.writerow([eval_entry_config["name"], experiment_paradigm, str(eval_digits), f"{avg_test_loss:.4f}", f"{accuracy:.2f}", correct, total, model_info])
    print(f"Results saved to {results_file}")
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the experiment YAML config file.")
    parser.add_argument('--eval_name', type=str, required=True, help="Evaluation name key from the config file.")
    parser.add_argument('--learning_paradigm', type=str, default="CIL", choices=["CIL", "TIL"], help="Learning paradigm: CIL or TIL.")
    args = parser.parse_args()

    abs_config_path = args.config
    if not os.path.isabs(abs_config_path):
        abs_config_path = os.path.join(project_root, args.config)
        
    evaluate_task(abs_config_path, args.eval_name, args.learning_paradigm)