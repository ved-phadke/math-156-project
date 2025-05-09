import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

class MNISTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128*3*3, 625)
        self.fc2 = nn.Linear(625, num_classes)

        self.act = nn.ReLU()
        self.pooling = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Layer 1: Conv -> ReLU -> MaxPool -> Dropout
        x = self.dropout(self.pooling(self.act(self.conv1(x))))

        # Layer 2: Conv -> ReLU -> MaxPool -> Dropout
        x = self.dropout(self.pooling(self.act(self.conv2(x))))

        # Layer 3: Conv -> ReLU -> MaxPool
        x = self.pooling(self.act(self.conv3(x)))

        x = torch.flatten(x, 1)
        
        # Layer 4: FC
        x = self.dropout(self.fc1(x))

        # Layer 5: FC
        x = self.fc2(x)

        return x

def get_task_datasets(digits):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_data = datasets.MNIST(root='./data', train=True, download=False, transform=data_transform)
    test_data = datasets.MNIST(root='./data', train=False, download=False, transform=data_transform)

    # Create a mapping from original labels to the new labels (0 - (len(digits)-1))
    label_mapping = {digit: idx for idx, digit in enumerate(digits)}

    train_indices = [i for i, (_, label) in enumerate(train_data) if label in digits]
    test_indices = [i for i, (_, label) in enumerate(test_data) if label in digits]

    def remap_labels(dataset, indices):
        data = []
        for i in indices:
            image, label = dataset[i]
            new_label = label_mapping[label]
            data.append((image, new_label))
        return data

    train_task_data = remap_labels(train_data, train_indices)
    test_task_data = remap_labels(test_data, test_indices)

    train_task = torch.utils.data.TensorDataset(
        torch.stack([item[0] for item in train_task_data]),
        torch.tensor([item[1] for item in train_task_data])
    )

    test_task = torch.utils.data.TensorDataset(
        torch.stack([item[0] for item in test_task_data]),
        torch.tensor([item[1] for item in test_task_data])
    )

    return DataLoader(train_task, batch_size=128, shuffle=True), DataLoader(test_task, batch_size=256, shuffle=True)
