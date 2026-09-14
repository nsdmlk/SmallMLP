import torch


def huber(e, delta=1.0):
    """Huber loss, element-wise. delta defaults to 1 (y is standardized)."""
    abs_e = torch.abs(e)
    quad = torch.clamp(abs_e, max=delta)
    lin = abs_e - quad
    return 0.5 * quad ** 2 + delta * lin


def loo_huber_loss(psi, X_train, y_train, forward_fn):
    """Leave-one-out Huber loss on the training set.

    psi:        (d,) unconstrained; h = softplus(psi)
    X_train:    (n, d)
    y_train:    (n,)
    forward_fn: callable(x_query, X_train, y_train, h, loo) -> (y_hat, delta, n_eff)
    """
    h = torch.nn.functional.softplus(psi)
    y_hat, _, _ = forward_fn(X_train, X_train, y_train, h, True)
    return huber(y_train - y_hat).mean()