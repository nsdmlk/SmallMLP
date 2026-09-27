import numpy as np
import torch
import torch.nn as nn


def adaptive_width(n, d, K=2, l=1):
    """Formula-based width for classification.

    w_l = max(K, min(floor(sqrt(K) * log2(n) * d / l), w_max))
    where w_max = min(4n, max(256, 32K)).
    """
    w_max = min(4 * n, max(256, 32 * K))
    raw = np.sqrt(K) * np.log2(max(n, 2)) * (d / max(l, 1))
    return int(max(K, min(np.floor(raw), w_max)))


def classic_width(n):
    """Original width: min(2n, 128)."""
    return int(min(max(2 * n, 16), 128))


def adaptive_dropout(n, d, p_max=0.5):
    return float(min(max(d / max(n, 1), 0.0), p_max))


class _Backbone(nn.Module):
    """Two-layer MLP with configurable width mode.

    width_mode:
      'formula'  — adaptive_width (default, for classification)
      'classic'  — min(2n, 128) for all layers (for regression)
    """

    def __init__(self, d_in, n, K=2, dropout=0.0, activation="relu",
                 n_layers=2, width_mode="formula"):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]

        if width_mode == "formula":
            widths = [
                adaptive_width(n, d_in, K=K, l=l)
                for l in range(1, n_layers + 1)
            ]
        elif width_mode == "classic":
            widths = [classic_width(n)] * n_layers
        else:
            raise ValueError(f"Unknown width_mode: {width_mode}")

        self.widths = widths
        self.width_mode = width_mode

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(d_in, widths[0]))
        for l in range(1, n_layers):
            self.layers.append(nn.Linear(widths[l - 1], widths[l]))

        self.acts = nn.ModuleList([act() for _ in range(n_layers)])
        self.drops = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])

        self.output_dim = widths[-1]

    def forward(self, x):
        h = x
        for fc, act, drop in zip(self.layers, self.acts, self.drops):
            h = drop(act(fc(h)))
        return h