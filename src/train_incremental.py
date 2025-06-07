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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import BaselineClassifier
from src.data_loader import get_filtered_mnist_dataloaders

def make_optimizer(model, optim_name, lr, alpha=None):
    """Factory for optimizers."""
    lr = float(lr)
    name = optim_name.lower()
    if name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, alpha=(alpha or 0.9))
    elif name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer {optim_name}")

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="YAML file (e.g. sharp_curvy_mnist.yaml)")
    args = parser.parse_args()

    # load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[ i ] Device: {device}")

    tasks = [(k, cfg[k]) for k in cfg if k.startswith('task')]
    tasks.sort(key=lambda kv: kv[0])  # ensure task1, task2, …

    base_model_dir = cfg.get('base_model_dir', 'models')
    os.makedirs(base_model_dir, exist_ok=True)

    # ------------ Stage 1: initial training ------------
    key1, t1 = tasks[0]
    digits1 = t1['digits']
    nc1 = len(digits1)
    print(f"[ 1 ] Initial training on digits {digits1} → head size = {nc1}")

    model = BaselineClassifier(num_classes=nc1).to(device)

    # data loader for initial digits
    train_loader, _ = get_filtered_mnist_dataloaders(
        digits=digits1,
        batch_size_train=t1['train_params']['batch_size'],
        data_root=cfg.get('data_dir', './data'),
        remap_labels=False  # labels already [0..nc1-1]
    )

    optimizer = make_optimizer(
        model,
        t1['train_params']['optimizer'],
        t1['train_params']['lr'],
        t1['train_params'].get('alpha', None)
    )
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(1, t1['train_params']['epochs']+1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Epoch {e}/{t1['train_params']['epochs']}  loss={loss:.4f}")

    # save initial model
    save1 = os.path.join(base_model_dir, t1['model_save_name'])
    torch.save(model.state_dict(), save1)
    print(f"[ ✓ ] Saved stage1 model to {save1}\n")

    # ------------ Subsequent stages ------------
    seen_digits = set(digits1)
    for stage, (key, task_cfg) in enumerate(tasks[1:], start=2):
        new_digits = task_cfg['digits']
        k = len(new_digits)
        print(f"[ {stage} ] Adding classes {new_digits}  (k={k})")

        # expand head
        model.add_classes(k)
        model.to(device)

        # rebuild optimizer so it picks up new params
        optimizer = make_optimizer(
            model,
            task_cfg['train_params']['optimizer'],
            task_cfg['train_params']['lr'],
            task_cfg['train_params'].get('alpha', None)
        )

        # train *only* on new digits (labels come in as natural MNIST labels)
        train_loader, _ = get_filtered_mnist_dataloaders(
            digits=new_digits,
            batch_size_train=task_cfg['train_params']['batch_size'],
            data_root=cfg.get('data_dir', './data'),
            remap_labels=False
        )

        for e in range(1, task_cfg['train_params']['epochs']+1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"  Epoch {e}/{task_cfg['train_params']['epochs']}  loss={loss:.4f}")

        # save
        save_path = os.path.join(base_model_dir, task_cfg['model_save_name'])
        torch.save(model.state_dict(), save_path)
        print(f"[ ✓ ] Saved stage{stage} model to {save_path}\n")

        # update seen set
        seen_digits.update(new_digits)

    print("[Done] All incremental stages complete.")

if __name__ == '__main__':
    main()