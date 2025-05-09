import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

def get_mnist_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5))
    ])

def get_full_mnist_dataloaders(batch_size_train=128, batch_size_test=256, data_root='../data'):
    """
    Gets DataLoaders for the full MNIST dataset.
    Assumes data_root is relative to the src/ directory if script is run from src,
    or relative to project root if script is run from project root.
    The paths in train.py and evaluate.py are set up assuming execution from project root.
    """
    data_transform = get_mnist_transforms()
    # Path adjusted for running scripts from project root (e.g. python src/train.py)
    # so data_root should be './data' or '../data' if data_loader.py is called from src/
    train_data = datasets.MNIST(root=data_root, train=True, download=True, transform=data_transform)
    test_data = datasets.MNIST(root=data_root, train=False, download=True, transform=data_transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=False)
    return train_loader, test_loader

def get_filtered_mnist_dataloaders(digits, batch_size_train=128, batch_size_test=256, data_root='../data'):
    """
    Gets DataLoaders for MNIST filtered by a list of specified digits.
    Pathing considerations similar to get_full_mnist_dataloaders.
    """
    data_transform = get_mnist_transforms()
    
    # Training data
    train_dataset_full = datasets.MNIST(root=data_root, train=True, download=True, transform=data_transform)
    train_indices = [i for i, (_, label) in enumerate(train_dataset_full) if label in digits]
    train_subset = Subset(train_dataset_full, train_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size_train, shuffle=True)
    
    # Test data
    test_dataset_full = datasets.MNIST(root=data_root, train=False, download=True, transform=data_transform)
    test_indices = [i for i, (_, label) in enumerate(test_dataset_full) if label in digits]
    test_subset = Subset(test_dataset_full, test_indices)
    test_loader = DataLoader(test_subset, batch_size=batch_size_test, shuffle=False)
    
    return train_loader, test_loader