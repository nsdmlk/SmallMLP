import torch

from .kernels import gaussian_weights, effective_sample_size


def soft_median_step(w, y_train, n_eff, alpha=1e-3):
    n = y_train.shape[0]
    sw = w.sum(dim=1) + 1e-12

    sigma_y2 = y_train.var(unbiased=False)

    y0 = (w * y_train[None, :]).sum(dim=1) / sw

    prior_w = (1.0 - n_eff / n).clamp(min=0.0, max=1.0)

    resid0 = y_train[None, :] - y0[:, None]
    num0 = (w * resid0 ** 2).sum(dim=1) + alpha * prior_w * sigma_y2
    den0 = sw + alpha * prior_w
    delta0 = torch.sqrt(num0 / den0 + 1e-12)

    tau = delta0 / torch.sqrt(n_eff + 1e-12)

    logits = -torch.abs(resid0) / (tau[:, None] + 1e-12)
    rw = w * torch.exp(logits)                                  # raw, not normalized
    rw_sum = rw.sum(dim=1) + 1e-12

    y_hat = (rw * y_train[None, :]).sum(dim=1) / rw_sum

    num = (rw * (y_train[None, :] - y_hat[:, None]) ** 2).sum(dim=1) \
          + alpha * prior_w * sigma_y2
    den = rw_sum + alpha * prior_w
    delta = torch.sqrt(num / den + 1e-12)

    return y_hat, delta

def forward(x_query, X_train, y_train, h, loo=False, alpha=1e-3):
    w = gaussian_weights(x_query, X_train, h)
    if loo:
        from .kernels import mask_self
        w = mask_self(w)
    n_eff = effective_sample_size(w)
    y_hat, delta = soft_median_step(w, y_train, n_eff, alpha=alpha)
    return y_hat, delta, n_eff