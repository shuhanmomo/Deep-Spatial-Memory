import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self, args):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(args["feature_size"], 64)  # First layer
        self.fc2 = nn.Linear(64, 32)  # Second layer
        self.fc3 = nn.Linear(32, ["label_num"])  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
