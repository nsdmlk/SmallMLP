import numpy as np
import torch


def weighted_quantile(values, weights, q):
    """Weighted quantile of `values` with `weights`."""
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


def _pairwise_sqdist(x_query, X_ref):
    """Squared Euclidean distance matrix, no (m, n, d) tensor."""
    xq_sq = (x_query ** 2).sum(dim=1, keepdim=True)
    xr_sq = (X_ref ** 2).sum(dim=1).unsqueeze(0)
    dist2 = xq_sq + xr_sq - 2.0 * (x_query @ X_ref.T)
    return torch.clamp(dist2, min=0.0)


def _gaussian_weights(x_query, X_ref, h):
    """Gaussian weights with vector bandwidth h (d,)."""
    xq = x_query / h[None, :]
    xr = X_ref / h[None, :]
    dist2 = _pairwise_sqdist(xq, xr)
    return torch.exp(-0.5 * dist2)


def _effective_sample_size(w):
    sw = w.sum(dim=1)
    sw2 = (w ** 2).sum(dim=1)
    return (sw ** 2) / (sw2 + 1e-12)


def _get_embedding(model, X):
    """Extract backbone embedding for X (n, width)."""
    X_t = model._prepare_query(X)
    with torch.no_grad():
        h = model._model(X_t)
    return h.numpy(), X_t


def conformal_qhat(x_query_emb, X_cal_emb, residuals_cal, h_cal, alpha):
    """Weighted conformal half-widths (regression)."""
    n_cal = X_cal_emb.shape[0]
    target_q = float(np.ceil((1 - alpha) * (n_cal + 1))) / n_cal
    target_q = min(target_q, 1.0)

    with torch.no_grad():
        w = _gaussian_weights(x_query_emb, X_cal_emb, h_cal)
        row_sums = w.sum(dim=1, keepdim=True)
        degenerate = (row_sums < 1e-12)
        if degenerate.any():
            w = torch.where(degenerate, torch.ones_like(w), w)
        n_eff = _effective_sample_size(w)
        w_norm = w / n_eff[:, None]

    res_np = residuals_cal.detach().numpy()
    w_np = w_norm.detach().numpy()
    q_hat = np.zeros(x_query_emb.shape[0], dtype=float)
    for i in range(x_query_emb.shape[0]):
        q_hat[i] = weighted_quantile(res_np, w_np[i], target_q)
    return q_hat


def conformal_qhat_classification(x_query_emb, X_cal_emb, scores_cal, h_cal, alpha):
    """Weighted conformal half-widths (classification scores)."""
    return conformal_qhat(x_query_emb, X_cal_emb, scores_cal, h_cal, alpha)


def tune_h_cal_regression(model, X_cal, y_cal, X_val, y_val,
                          alpha=0.1, h_grid=None):
    """Grid search h_cal for regression by coverage/width tradeoff."""
    if h_grid is None:
        h_grid = np.logspace(np.log10(0.1), np.log10(10.0), 25)

    y_cal_hat = model.predict(X_cal)
    residuals_cal = np.abs(y_cal - y_cal_hat)

    X_cal_emb, _ = _get_embedding(model, X_cal)
    X_val_emb, _ = _get_embedding(model, X_val)
    residuals_t = torch.tensor(residuals_cal, dtype=torch.float32)

    y_val_hat = model.predict(X_val)
    d = X_cal_emb.shape[1]

    coverage_target = 1.0 - alpha
    best = {"h_cal": None, "loss": np.inf, "width": None, "coverage": None}

    for h_val in h_grid:
        h_cal = torch.full((d,), float(h_val), dtype=torch.float32)
        q_hat = conformal_qhat(
            X_val_emb, X_cal_emb, residuals_t, h_cal, alpha
        )
        lo = y_val_hat - q_hat
        hi = y_val_hat + q_hat
        coverage = float(np.mean((y_val >= lo) & (y_val <= hi)))
        width = float(np.mean(hi - lo))

        if coverage < coverage_target - 0.02:
            loss = 1e9 + width
        else:
            loss = width

        if loss < best["loss"]:
            best = {"h_cal": float(h_val), "loss": loss,
                    "width": width, "coverage": coverage}
    return best

def tune_h_cal_classification(model, X_cal, y_cal, X_val, y_val,
                              alpha=0.1, h_grid=None):
    """Grid search h_cal for classification by coverage/set-size tradeoff."""
    if h_grid is None:
        h_grid = np.logspace(np.log10(0.1), np.log10(10.0), 25)

    y_cal_enc = model._label_encoder.transform(y_cal).astype(np.int64)
    p_cal = model.predict_proba(X_cal)
    scores_cal = 1.0 - p_cal[np.arange(len(y_cal_enc)), y_cal_enc]

    X_cal_emb, _ = _get_embedding(model, X_cal)
    X_val_emb, _ = _get_embedding(model, X_val)
    scores_t = torch.tensor(scores_cal, dtype=torch.float32)

    p_val = model.predict_proba(X_val)
    y_val_enc = model._label_encoder.transform(y_val).astype(np.int64)
    d = X_cal_emb.shape[1]

    coverage_target = 1.0 - alpha
    best = {"h_cal": None, "loss": np.inf, "size": None, "coverage": None}

    for h_val in h_grid:
        h_cal = torch.full((d,), float(h_val), dtype=torch.float32)
        q_hat = conformal_qhat(
            X_val_emb, X_cal_emb, scores_t, h_cal, alpha
        )
        sizes = np.zeros(len(X_val))
        covered = np.zeros(len(X_val), dtype=bool)
        for i in range(len(X_val)):
            c = 0
            for k in range(model.n_classes_):
                if 1.0 - p_val[i, k] <= q_hat[i]:
                    c += 1
            if c == 0:
                c = 1
            sizes[i] = c
            covered[i] = 1.0 - p_val[i, y_val_enc[i]] <= q_hat[i]
            if not covered[i] and c == 1:
                # fallback: include argmax; coverage may still fail
                pass

        coverage = float(covered.mean())
        size = float(sizes.mean())

        if coverage < coverage_target - 0.02:
            loss = 1e9 + size
        else:
            loss = size

        if loss < best["loss"]:
            best = {"h_cal": float(h_val), "loss": loss,
                    "size": size, "coverage": coverage}
    return best