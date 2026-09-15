import torch


def gaussian_weights(x_query, X_train, h):
    """Gaussian kernel weights with vector bandwidth.

    Uses ||x - x_i||^2 = ||x||^2 - 2 x . x_i + ||x_i||^2
    to avoid materializing the (m, n, d) diff tensor.

    x_query: (m, d) tensor
    X_train: (n, d) tensor
    h:       (d,) tensor, positive
    returns: (m, n) tensor
    """
    # scaled coordinates: divide by h, then use the expansion
    xq = x_query / h[None, :]                                  # (m, d)
    xt = X_train / h[None, :]                                  # (n, d)

    xq_sq = (xq ** 2).sum(dim=1, keepdim=True)                 # (m, 1)
    xt_sq = (xt ** 2).sum(dim=1).unsqueeze(0)                  # (1, n)

    # (m, n) = (m,1) + (1,n) - 2 * xq @ xt^T
    dist2 = xq_sq + xt_sq - 2.0 * (xq @ xt.T)

    # numerical clamp: dist2 can be slightly negative due to float error
    dist2 = torch.clamp(dist2, min=0.0)

    return torch.exp(-0.5 * dist2)


def effective_sample_size(w):
    """n_eff = (sum w)^2 / sum w^2, per row.

    w: (m, n) tensor
    returns: (m,) tensor
    """
    sw = w.sum(dim=1)
    sw2 = (w ** 2).sum(dim=1)
    return (sw ** 2) / (sw2 + 1e-12)


def mask_self(w):
    """Zero out the diagonal (leave-one-out). w: (n, n) square tensor."""
    n = w.shape[0]
    mask = torch.eye(n, dtype=torch.bool, device=w.device)
    return w.masked_fill(mask, 0.0)