# Adapted from: https://github.com/connorlee77/pytorch-xcorr2/blob/master/xcorr2.py

import torch
import torch.nn as nn


class XCorr2(nn.Module):
    """
    Compute the normalized cross-correlation between two images with the same shape.
    """

    def __init__(self, zero_mean_normalized=False):
        super(XCorr2, self).__init__()
        self.InstanceNorm = nn.InstanceNorm2d(1)
        self.zero_mean_normalized = zero_mean_normalized

    def forward(self, x1, x2):
        assert x1.shape == x2.shape
        _, h, w = x1.shape
        if self.zero_mean_normalized:
            x1 = self.InstanceNorm(x1.expand(1, -1, -1))
            x2 = self.InstanceNorm(x2.expand(1, -1, -1))
        score = torch.einsum("b...,b...->b", x1, x2)
        score /= h * w
        return score
