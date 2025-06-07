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
    
    def add_classes(self, k):
        """
        Expand the final linear by k new outputs
        """
        W_old = self.fc2.weight.data
        b_old = self.fc2.bias.data
        old_nc, feat_dim = W_old.shape

        # make a new head with _old_nc + k outputs
        new_head = nn.Linear(feat_dim, old_nc + k)

        with torch.no_grad():
            # copy existing weights/biases
            new_head.weight[:old_nc].copy_(W_old)
            new_head.bias[:old_nc].copy_(b_old)
            # new_head.weight[old_nc:] is freshly initialized
        
        self.fc2 = new_head