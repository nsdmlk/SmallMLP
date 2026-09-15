import torch

from .kernels import gaussian_weights, effective_sample_size


def weighted_mean_step(w, y_train, n_eff, alpha=1e-3):
    """Weighted mean + distance-aware zone.

    w:       (m, n) kernel weights
    y_train: (n,) tensor
    n_eff:   (m,) effective sample size
    alpha:   prior strength (small, dimensionless)
    returns: y_hat (m,), delta (m,)
    """
    n = y_train.shape[0]
    sw = w.sum(dim=1) + 1e-12

    # global prior on variance
    sigma_y2 = y_train.var(unbiased=False)

    # weighted mean
    y_hat = (w * y_train[None, :]).sum(dim=1) / sw

    # distance-aware prior weight: 0 when n_eff = n, 1 when n_eff -> 0
    prior_w = (1.0 - n_eff / n).clamp(min=0.0, max=1.0)

    # zone: weighted variance, interpolated with global prior
    resid = y_train[None, :] - y_hat[:, None]
    num = (w * resid ** 2).sum(dim=1) + alpha * prior_w * sigma_y2
    den = sw + alpha * prior_w
    delta = torch.sqrt(num / den + 1e-12)

    return y_hat, delta


def forward(x_query, X_train, y_train, h, loo=False, alpha=1e-3):
    w = gaussian_weights(x_query, X_train, h)
    if loo:
        from .kernels import mask_self
        w = mask_self(w)
    n_eff = effective_sample_size(w)
    y_hat, delta = weighted_mean_step(w, y_train, n_eff, alpha=alpha)
    return y_hat, delta, n_eff