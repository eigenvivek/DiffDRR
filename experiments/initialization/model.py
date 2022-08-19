import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, dtype=torch.float32, device="cuda"):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5, dtype=dtype, device=device)
        self.conv2 = torch.nn.Conv2d(6, 16, 3, dtype=dtype, device=device)
        self.fc1 = torch.nn.Linear(16 * 23 * 23, 120, dtype=dtype, device=device)
        self.fc2 = torch.nn.Linear(120, 84, dtype=dtype, device=device)
        self.fc3 = torch.nn.Linear(84, 6, dtype=dtype, device=device)

    def forward(self, x):
        x = x.view(-1, 1, 100, 100)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        return torch.tensor(x.shape[1:]).prod().item()
