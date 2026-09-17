import numpy as np
import torch

from .kernels import gaussian_weights, effective_sample_size


def weighted_quantile(values, weights, q):
    """Weighted quantile of `values` with `weights`.

    values:  (n,) array
    weights: (n,) non-negative array
    q:       target quantile in (0, 1)
    returns: float
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        return float(np.quantile(values, q))
    idx = np.argsort(values)
    v_sorted = values[idx]
    w_sorted = weights[idx]
    cw = np.cumsum(w_sorted) / w_sorted.sum()
    pos = int(np.searchsorted(cw, q, side="left"))
    pos = min(pos, len(v_sorted) - 1)
    return float(v_sorted[pos])


def conformal_qhat(x_query, X_cal, residuals_cal, h_cal, alpha):
    n_cal = X_cal.shape[0]
    target_q = float(np.ceil((1 - alpha) * (n_cal + 1))) / n_cal
    target_q = min(target_q, 1.0)

    with torch.no_grad():
        w = gaussian_weights(x_query, X_cal, h_cal)     # (m, n_cal)
        # safety: if any row is all-zero (degenerate h), fall back to uniform
        row_sums = w.sum(dim=1, keepdim=True)
        degenerate = (row_sums < 1e-12)
        if degenerate.any():
            w = torch.where(degenerate, torch.ones_like(w), w)
        n_eff = effective_sample_size(w)
        w_norm = w / n_eff[:, None]

    res_np = residuals_cal.detach().numpy()
    w_np = w_norm.detach().numpy()
    q_hat = np.zeros(x_query.shape[0], dtype=float)
    for i in range(x_query.shape[0]):
        q_hat[i] = weighted_quantile(res_np, w_np[i], target_q)
    return q_hat


def tune_h_cal(model, X_train, y_train, X_cal, y_cal, X_val, y_val,
               alpha=0.05, h_grid=None, lambda_penalty=10.0):
    """Grid search h_cal by validation coverage/width tradeoff.

    Rejects candidates whose coverage is more than 2% below target,
    then picks the narrowest valid interval.
    """
    if h_grid is None:
        # lower bound 0.5 avoids degenerate weights under float64
        h_grid = np.logspace(np.log10(0.5), np.log10(10.0), 20)

    y_cal_hat = model.predict(X_cal)
    residuals_cal = np.abs(y_cal - y_cal_hat)

    X_cal_t = _to_tensor(model, X_cal)
    X_val_t = _to_tensor(model, X_val)
    residuals_t = torch.tensor(residuals_cal, dtype=torch.float64)

    y_val_hat = model.predict(X_val)
    d = X_cal.shape[1]

    coverage_target = 1.0 - alpha
    best = {"h_cal": None, "loss": np.inf, "width": None, "coverage": None}

    for h_val in h_grid:
        h_cal = torch.full((d,), float(h_val), dtype=torch.float64)
        q_hat = conformal_qhat(
            X_val_t, X_cal_t, residuals_t, h_cal, alpha
        )
        lo = y_val_hat - q_hat
        hi = y_val_hat + q_hat
        coverage = float(np.mean((y_val >= lo) & (y_val <= hi)))
        width = float(np.mean(hi - lo))

        # reject candidates well below target coverage
        if coverage < coverage_target - 0.02:
            loss = 1e9 + width
        else:
            loss = width

        if loss < best["loss"]:
            best = {
                "h_cal": float(h_val),
                "loss": loss,
                "width": width,
                "coverage": coverage,
            }

    return best

def conformal_qhat_classification(x_query, X_cal, scores_cal, h_cal, alpha):
    """Weighted conformal quantile for classification scores.

    x_query:    (m, d) tensor
    X_cal:      (n_cal, d) tensor
    scores_cal: (n_cal,) tensor of conformal scores
    h_cal:      (d,) tensor
    alpha:      miscoverage
    returns:    (m,) numpy array of q_hat
    """
    n_cal = X_cal.shape[0]
    target_q = float(np.ceil((1 - alpha) * (n_cal + 1))) / n_cal
    target_q = min(target_q, 1.0)

    with torch.no_grad():
        w = gaussian_weights(x_query, X_cal, h_cal)
        row_sums = w.sum(dim=1, keepdim=True)
        degenerate = (row_sums < 1e-12)
        if degenerate.any():
            w = torch.where(degenerate, torch.ones_like(w), w)
        n_eff = effective_sample_size(w)
        w_norm = w / n_eff[:, None]

    scores_np = scores_cal.detach().numpy()
    w_np = w_norm.detach().numpy()
    q_hat = np.zeros(x_query.shape[0], dtype=float)
    for i in range(x_query.shape[0]):
        q_hat[i] = weighted_quantile(scores_np, w_np[i], target_q)
    return q_hat


def tune_h_cal_classification(model, X_cal, y_cal_bin, X_val, y_val_bin,
                              alpha=0.1, h_grid=None):
    """Grid search h_cal by validation coverage/set-size tradeoff.

    y_cal_bin, y_val_bin: binary targets in {0, 1}.
    """
    if h_grid is None:
        h_grid = np.logspace(np.log10(0.5), np.log10(10.0), 20)

    p_cal = model._predict_p(X_cal)
    scores_cal = np.where(y_cal_bin == 1, 1.0 - p_cal, p_cal)

    X_cal_t = _to_tensor(model, X_cal)
    X_val_t = _to_tensor(model, X_val)
    scores_t = torch.tensor(scores_cal, dtype=torch.float64)

    p_val = model._predict_p(X_val)
    d = X_cal.shape[1]

    coverage_target = 1.0 - alpha
    best = {"h_cal": None, "loss": np.inf, "size": None, "coverage": None}

    for h_val in h_grid:
        h_cal = torch.full((d,), float(h_val), dtype=torch.float64)
        q_hat = conformal_qhat_classification(
            X_val_t, X_cal_t, scores_t, h_cal, alpha
        )

        sizes = np.zeros(len(X_val))
        covered = np.zeros(len(X_val), dtype=bool)
        for i in range(len(X_val)):
            s_0 = p_val[i]
            s_1 = 1.0 - p_val[i]
            c0 = s_0 <= q_hat[i]
            c1 = s_1 <= q_hat[i]
            if not c0 and not c1:
                # fallback: include more likely class
                if s_0 < s_1:
                    c0 = True
                else:
                    c1 = True
            sizes[i] = int(c0) + int(c1)
            covered[i] = (y_val_bin[i] == 0 and c0) or \
                         (y_val_bin[i] == 1 and c1)

        coverage = float(covered.mean())
        size = float(sizes.mean())

        if coverage < coverage_target - 0.02:
            loss = 1e9 + size
        else:
            loss = size

        if loss < best["loss"]:
            best = {
                "h_cal": float(h_val),
                "loss": loss,
                "size": size,
                "coverage": coverage,
            }

    return best

def _to_tensor(model, X):
    Xs = (X - model._x_mean) / model._x_std
    return torch.tensor(Xs, dtype=torch.float64)