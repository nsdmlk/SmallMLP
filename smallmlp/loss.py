import torch


def huber(e, delta=1.0):
    """Huber loss, element-wise. delta defaults to 1 (y is standardized)."""
    abs_e = torch.abs(e)
    quad = torch.clamp(abs_e, max=delta)
    lin = abs_e - quad
    return 0.5 * quad ** 2 + delta * lin


def loo_huber_loss(h, X_train, y_train, forward_fn):
    """Leave-one-out Huber loss on the training set.

    h:          (d,) bandwidth vector (already mapped from psi)
    X_train:    (n, d)
    y_train:    (n,)
    forward_fn: callable(x_query, X_train, y_train, h, loo) -> (y_hat, delta, n_eff)
    """
    y_hat, _, _ = forward_fn(X_train, X_train, y_train, h, True)
    return huber(y_train - y_hat).mean()

def loo_bce_loss(h, X_train, y_train, forward_fn, eps=1e-6):
    """Leave-one-out BCE loss (for classifier training).

    y_train in {0, 1}. p_hat from forward is clipped for numerical stability.
    """
    p_hat, _, _ = forward_fn(X_train, X_train, y_train, h, True)
    p_hat = torch.clamp(p_hat, eps, 1.0 - eps)
    return -(y_train * torch.log(p_hat) + (1 - y_train) * torch.log(1 - p_hat)).mean()