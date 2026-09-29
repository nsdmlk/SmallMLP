import numpy as np
import torch
import torch.nn as nn


def adaptive_width(n, d, K=2, l=1, alpha=4.0, beta=0.7):
    """Adaptive width for classification.

    First layer:
      w_1 = max(2K, min(floor(alpha * sqrt(n * K * sqrt(d))), 4n))

    Subsequent layers:
      w_l = max(K, floor(w_1 * beta^(l-1)))

    Properties:
      - Grows sublinearly with n, K (as sqrt), and d (as d^(1/4)).
      - First layer bounded below by 2K (one unit per class + spare).
      - Later layers bounded below by K, decayed by beta.
      - Capped at 4n to prevent blow-up on tiny datasets.

    alpha=4.0 default — chosen by empirical sweep (see benchmarks/).
    beta=0.7 — fixed decay between layers.
    """
    w_max = 4 * n
    K_lower_first = 2 * K if K > 2 else K

    raw = alpha * np.sqrt(n * K * np.sqrt(max(d, 1)))
    w1 = int(max(K_lower_first, min(np.floor(raw), w_max)))

    if l == 1:
        return w1

    wl = int(max(K, np.floor(w1 * (beta ** (l - 1)))))
    return wl


def classic_width(n):
    """Original width: min(2n, 128)."""
    return int(min(max(2 * n, 16), 128))


def adaptive_dropout(n, d, c=3.0, p_min=0.1, p_max=0.5):
    """Clamped adaptive dropout: p = clamp(c * d/n, p_min, p_max)."""
    return float(min(max(c * d / max(n, 1), p_min), p_max))


# ----------------------------------------------------------------------
# Custom activations
# ----------------------------------------------------------------------

class AlgSig(nn.Module):
    """x / sqrt(1 + x^2) — algebraic sigmoid.

    Odd, smooth, saturating to ±1, linear near origin (sigma'(0) = 1).
    No parameters, faster than tanh (no exp), retains more information
    than tanh on N(0,1) inputs.
    """
    def forward(self, x):
        return x / torch.sqrt(1.0 + x * x)


class SoftSign(nn.Module):
    """x / (1 + |x|) — softsign.

    Similar to AlgSig but with heavier tails.
    """
    def forward(self, x):
        return x / (1.0 + x.abs())


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "algsig": AlgSig,
    "softsign": SoftSign,
}


class _Backbone(nn.Module):
    """Two-layer MLP with configurable width mode.

    width_mode:
      'formula'  — adaptive_width (default, for classification)
      'classic'  — min(2n, 128) for all layers (for regression)

    activation:
      'relu' | 'tanh' | 'gelu' | 'silu' | 'algsig' | 'softsign'
    """

    def __init__(self, d_in, n, K=2, dropout=0.0, activation="relu",
                 n_layers=2, width_mode="formula", alpha=4.0, beta=0.7):
        super().__init__()

        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation: {activation!r}. "
                f"Available: {sorted(_ACTIVATIONS.keys())}"
            )
        act = _ACTIVATIONS[activation]

        if width_mode == "formula":
            widths = [
                adaptive_width(n, d_in, K=K, l=l, alpha=alpha, beta=beta)
                for l in range(1, n_layers + 1)
            ]
        elif width_mode == "classic":
            widths = [classic_width(n)] * n_layers
        else:
            raise ValueError(f"Unknown width_mode: {width_mode}")

        self.widths = widths
        self.width_mode = width_mode
        self.activation = activation

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