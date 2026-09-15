import torch


def huber(e, delta=1.0):
    """Huber loss, element-wise. delta defaults to 1 (y is standardized)."""
    abs_e = torch.abs(e)
    quad = torch.clamp(abs_e, max=delta)
    lin = abs_e - quad
    return 0.5 * quad ** 2 + delta * lin


def loo_huber_loss(h, X_train, y_train, forward_fn):
    """Leave-one-out Huber loss on the training set."""
    y_hat, _, _ = forward_fn(X_train, X_train, y_train, h, True)
    return huber(y_train - y_hat).mean()