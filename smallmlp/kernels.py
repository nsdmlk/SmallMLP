import torch


def gaussian_weights(x_query, X_train, h):
    """Gaussian kernel weights with vector bandwidth.

    x_query: (m, d) tensor
    X_train: (n, d) tensor
    h:       (d,) tensor, positive
    returns: (m, n) tensor
    """
    diff = x_query[:, None, :] - X_train[None, :, :]      # (m, n, d)
    dist2 = ((diff ** 2) / (h ** 2)[None, None, :]).sum(dim=-1)
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