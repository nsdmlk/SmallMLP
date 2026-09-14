import torch

from .kernels import gaussian_weights, effective_sample_size


def soft_median_step(w, y_train, n_eff):
    """One IRLS step of weighted soft-median, with automatic temperature.

    w:       (m, n) kernel weights
    y_train: (n,) tensor
    n_eff:   (m,) effective sample size
    returns: y_hat (m,), delta (m,)
    """
    sw = w.sum(dim=1) + 1e-12

    # initial weighted mean
    y0 = (w * y_train[None, :]).sum(dim=1) / sw                # (m,)

    # local scale before reweighting
    resid0 = y_train[None, :] - y0[:, None]                    # (m, n)
    delta0 = torch.sqrt(
        (w * resid0 ** 2).sum(dim=1) / sw + 1e-12
    )                                                          # (m,)

    # automatic temperature: tau = delta0 / sqrt(n_eff)
    tau = delta0 / torch.sqrt(n_eff + 1e-12)                   # (m,)

    # one reweighting step
    logits = -torch.abs(resid0) / (tau[:, None] + 1e-12)
    rw = w * torch.exp(logits)
    rw = rw / (rw.sum(dim=1, keepdim=True) + 1e-12)            # (m, n)

    y_hat = (rw * y_train[None, :]).sum(dim=1)                 # (m,)

    # zone: weighted std around y_hat
    delta = torch.sqrt(
        (rw * (y_train[None, :] - y_hat[:, None]) ** 2).sum(dim=1) + 1e-12
    )
    return y_hat, delta


def forward(x_query, X_train, y_train, h, loo=False):
    """Full SmallMLP forward.

    x_query: (m, d), X_train: (n, d), y_train: (n,), h: (d,)
    loo: if True, mask diagonal (requires m == n and x_query is X_train)
    returns: y_hat (m,), delta (m,), n_eff (m,)
    """
    w = gaussian_weights(x_query, X_train, h)
    if loo:
        from .kernels import mask_self
        w = mask_self(w)
    n_eff = effective_sample_size(w)
    y_hat, delta = soft_median_step(w, y_train, n_eff)
    return y_hat, delta, n_eff