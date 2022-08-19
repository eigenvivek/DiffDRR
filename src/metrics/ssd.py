import torch
import torch.nn as nn


class SSD(nn.Module):
    """
    Compute the sum of square differences (SSD).
    """

    def __init__(self):
        super(SSD, self).__init__()

    def forward(self, x1, x2):
        assert x1.shape == x2.shape
        return torch.sum(torch.square(x1 - x2))
