import numpy as np
import torch
import torch.nn as nn


def adaptive_width(n, d, w_max=128):
    return int(min(max(2 * n, 16), w_max))


def adaptive_dropout(n, d, p_max=0.5):
    return float(min(max(d / max(n, 1), 0.0), p_max))


class _Backbone(nn.Module):
    """Two-layer MLP with adaptive width and dropout."""

    def __init__(self, d_in, width, dropout, activation="relu"):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
        self.fc1 = nn.Linear(d_in, width)
        self.act1 = act()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(width, width)
        self.act2 = act()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop1(self.act1(self.fc1(x)))
        h = self.drop2(self.act2(self.fc2(h)))
        return h