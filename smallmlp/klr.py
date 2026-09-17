"""Kernel Logistic Regression internals for SmallMLPClassifier."""

import torch

from .kernels import gaussian_weights


def klr_forward(x_query, X_train, alpha, b, h, loo=False):
    """Kernel logistic regression forward.

    x_query:  (m, d) tensor
    X_train:  (n, d) tensor
    alpha:    (n,) tensor of coefficients
    b:        scalar tensor
    h:        (d,) tensor
    loo:      if True, mask diagonal (requires x_query is X_train)
    returns:  p_hat (m,) probabilities
    """
    K = gaussian_weights(x_query, X_train, h)   # (m, n)

    if loo:
        n = X_train.shape[0]
        mask = torch.eye(n, dtype=torch.bool, device=K.device)
        K = K.masked_fill(mask, 0.0)

    f = K @ alpha + b
    return torch.sigmoid(f)


def klr_loo_bce_loss(alpha, b, psi, X_train, y_train, h_min, h_max,
                     lam=1e-3, eps=1e-6, class_weight=None):
    """LOO-BCE + L2 regularization for KLR.

    class_weight: None or "balanced".
    """
    h = h_min + (h_max - h_min) * torch.sigmoid(psi)
    p_hat = klr_forward(X_train, X_train, alpha, b, h, loo=True)
    p_hat = torch.clamp(p_hat, eps, 1.0 - eps)

    if class_weight == "balanced":
        n = y_train.shape[0]
        n1 = y_train.sum()
        n0 = n - n1
        w1 = n / (2.0 * n1 + 1e-12)
        w0 = n / (2.0 * n0 + 1e-12)
        weights = y_train * w1 + (1 - y_train) * w0
    else:
        weights = torch.ones_like(y_train)

    bce = -(weights * (y_train * torch.log(p_hat)
                       + (1 - y_train) * torch.log(1 - p_hat))).mean()
    reg = lam * (alpha ** 2).sum()
    return bce + reg