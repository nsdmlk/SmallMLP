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


def loo_bce_loss(h, X_train, y_train, forward_fn, eps=1e-6,
                 class_weight=None):
    """Leave-one-out BCE loss for binary classification.

    y_train in {0, 1}. class_weight: None or "balanced".
    """
    p_hat, _, _ = forward_fn(X_train, X_train, y_train, h, True)
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

    return -(weights * (y_train * torch.log(p_hat)
                        + (1 - y_train) * torch.log(1 - p_hat))).mean()