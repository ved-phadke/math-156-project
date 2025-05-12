import torch
import torch.nn as nn

class BaselineClassifier(nn.Module):
    def __init__(self, num_classes=10): # Added num_classes, default to 10 for CIL
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128*3*3, 625) 
        self.fc2 = nn.Linear(625, num_classes) # Use num_classes

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