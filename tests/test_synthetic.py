import numpy as np
import pytest

from smallmlp import SmallMLPRegressor


def _fit(X, y, **kw):
    model = SmallMLPRegressor(h_init=1.0, max_iter=100, tol=1e-8, **kw)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# 1. Zone grows in empty regions: far from train data -> wider delta.
# ---------------------------------------------------------------------------

def test_zone_grows_in_empty_region():
    rng = np.random.default_rng(0)
    n, d = 150, 3
    # train cluster: tight gaussian around origin
    X_train = rng.normal(scale=0.3, size=(n, d))
    y_train = np.sin(5 * X_train[:, 0]) + 0.1 * rng.normal(size=n)

    model = _fit(X_train, y_train)

    # queries: one inside the cluster, one far away
    X_inside = rng.normal(scale=0.3, size=(20, d))
    X_far = rng.normal(scale=0.3, size=(20, d)) + 5.0  # shift far

    _, delta_inside = model.predict_zone(X_inside)
    _, delta_far = model.predict_zone(X_far)

    assert delta_far.mean() > delta_inside.mean(), (
        f"delta_far={delta_far.mean():.4f} not > delta_inside={delta_inside.mean():.4f}"
    )


# ---------------------------------------------------------------------------
# 2. Zone grows around outliers in the training set.
# ---------------------------------------------------------------------------

def test_zone_grows_around_outliers():
    rng = np.random.default_rng(1)
    n, d = 150, 3
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 2.0 + 0.1 * rng.normal(size=n)

    # two regions in x-space: A (clean) and B (with outliers)
    X_clean = X.copy()
    y_clean = y.copy()

    X_out = X.copy()
    y_out = y.copy()
    # inject outliers in a localized region: points near x[0] ~ +2
    out_mask = X[:, 0] > 2.0
    y_out[out_mask] += 50.0  # massive outliers

    model_clean = _fit(X_clean, y_clean)
    model_out = _fit(X_out, y_out)

    # query at the outlier region
    X_query = np.array([[2.5, 0.0, 0.0]])
    _, delta_clean = model_clean.predict_zone(X_query)
    _, delta_out = model_out.predict_zone(X_query)

    assert delta_out[0] > delta_clean[0], (
        f"delta_out={delta_out[0]:.4f} not > delta_clean={delta_clean[0]:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. Point prediction is robust to outliers: SmallMLP is more stable than
#    a plain weighted mean (which is what a naive kernel smoother does).
# ---------------------------------------------------------------------------

def _weighted_mean_predictor(X_train, y_train, X_query, h=1.0):
    """Naive kernel smoother: weighted mean, no reweighting, no median."""
    diff = X_query[:, None, :] - X_train[None, :, :]
    dist2 = (diff ** 2).sum(axis=-1) / (h ** 2)
    w = np.exp(-0.5 * dist2)
    return (w * y_train[None, :]).sum(axis=1) / w.sum(axis=1)


def test_point_is_robust_to_outliers():
    rng = np.random.default_rng(2)
    n, d = 200, 2
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 2.0 + 0.2 * rng.normal(size=n)

    # clean vs contaminated
    X_clean, y_clean = X.copy(), y.copy()
    X_out, y_out = X.copy(), y.copy()
    idx = rng.choice(n, size=10, replace=False)
    y_out[idx] += 100.0  # heavy contamination

    model_clean = _fit(X_clean, y_clean)
    model_out = _fit(X_out, y_out)

    # query points spread around
    X_query = rng.normal(size=(50, d))

    y_smallmlp_clean = model_clean.predict(X_query)
    y_smallmlp_out = model_out.predict(X_query)
    shift_smallmlp = np.abs(y_smallmlp_out - y_smallmlp_clean).mean()

    y_wm_clean = _weighted_mean_predictor(X_clean, y_clean, X_query, h=1.0)
    y_wm_out = _weighted_mean_predictor(X_out, y_out, X_query, h=1.0)
    shift_wm = np.abs(y_wm_out - y_wm_clean).mean()

    assert shift_smallmlp < shift_wm, (
        f"SmallMLP shift={shift_smallmlp:.4f} not < weighted-mean shift={shift_wm:.4f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])