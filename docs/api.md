
# API Reference

Complete reference for the `smallmlp` package.

---

## `SmallMLPRegressor`

```python
from smallmlp import SmallMLPRegressor
```

A scikit-learn compatible regressor implementing learned-bandwidth Nadaraya-Watson regression with weighted conformal prediction intervals.

### Constructor

```python
SmallMLPRegressor(
    h_min=0.01,
    h_max=10.0,
    max_iter=30,
    inner_iter=10,
    tol=1e-6,
    verbose=False,
    alpha=1e-3,
)
```

| Parameter      | Type  | Default   | Description                                                                                                                          |
| -------------- | ----- | --------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `h_min`      | float | `0.01`  | Lower bound for the bandwidth vector (in standardized coordinates). Prevents the kernel from collapsing to a point.                  |
| `h_max`      | float | `10.0`  | Upper bound for the bandwidth. Prevents the kernel from becoming flat (globally constant model).                                     |
| `max_iter`   | int   | `30`    | Maximum number of outer L-BFGS iterations.                                                                                           |
| `inner_iter` | int   | `10`    | Maximum number of inner L-BFGS iterations per outer step.                                                                            |
| `tol`        | float | `1e-6`  | Convergence tolerance for L-BFGS (both gradient and parameter change).                                                               |
| `verbose`    | bool  | `False` | If`True`, print loss at each outer iteration during `fit`.                                                                       |
| `alpha`      | float | `1e-3`  | Prior strength for the heuristic zone. Larger values pull$\delta(x)$ toward the global standard deviation faster in empty regions. |

**Note on hyperparameters.** The defaults are the values used in all benchmarks. Tuning is not expected to improve results significantly — the model is designed to work without it.

### Attributes (after `fit`)

| Attribute               | Type             | Description                                   |
| ----------------------- | ---------------- | --------------------------------------------- |
| `n_features_in_`      | int              | Number of features seen during`fit`.        |
| `classes_`            | —               | Not present (regressor only).                 |
| `_psi`                | `torch.Tensor` | Unconstrained bandwidth parameter (internal). |
| `_X_train`            | `torch.Tensor` | Standardized training features (internal).    |
| `_y_train`            | `torch.Tensor` | Standardized training targets (internal).     |
| `_loss_`              | float            | Final LOO-Huber loss value.                   |
| `_x_mean`, `_x_std` | `np.ndarray`   | Feature standardization statistics.           |
| `_y_mean`, `_y_std` | float            | Target standardization statistics.            |

### Attributes (after `fit_conformal`)

| Attribute            | Type           | Description                                     |
| -------------------- | -------------- | ----------------------------------------------- |
| `_h_cal`           | `np.ndarray` | Tuned calibration bandwidth vector.             |
| `_residuals_cal`   | `np.ndarray` | Calibration residuals in original target units. |
| `_X_cal_raw`       | `np.ndarray` | Raw calibration features.                       |
| `_alpha_conformal` | float          | Miscoverage level used during calibration.      |

---

## Methods

### `fit(X, y)`

Train the point predictor.

**Parameters:**

- `X` — array-like of shape `(n_samples, n_features)`.
- `y` — array-like of shape `(n_samples,)`.

**Returns:** `self`.

**Notes:**

- Features and targets are standardized internally.
- Bandwidth `h` is learned by minimizing leave-one-out Huber loss via L-BFGS.
- For $n < 500$, training completes in under one second.

**Example:**

```python
model = SmallMLPRegressor()
model.fit(X_train, y_train)
```

---

### `predict(X)`

Predict target values.

**Parameters:**

- `X` — array-like of shape `(n_samples, n_features)`.

**Returns:** `np.ndarray` of shape `(n_samples,)`.

**Notes:**

- Returns kernel-weighted mean at each query point.
- Complexity: $O(q \cdot n \cdot d)$ for $q$ queries.

**Example:**

```python
y_hat = model.predict(X_test)
```

---

### `predict_zone(X)`

Return point predictions and heuristic zone half-widths.

**Parameters:**

- `X` — array-like of shape `(n_samples, n_features)`.

**Returns:** tuple `(y_hat, delta)` of `np.ndarray` of shape `(n_samples,)`.

**Notes:**

- `y_hat` — point prediction (same as `predict`).
- `delta` — heuristic zone half-width, computed as the distance-aware weighted standard deviation (see [Method](method.md) §3).
- **No coverage guarantee.** For calibrated intervals use `predict_interval_conformal`.

**Example:**

```python
y_hat, delta = model.predict_zone(X_test)
```

---

### `predict_interval(X, alpha=0.95)`

Return heuristic Gaussian-style interval.

**Parameters:**

- `X` — array-like of shape `(n_samples, n_features)`.
- `alpha` — float in `(0, 1)`. Coverage level. Default `0.95`.

**Returns:** tuple `(lower, upper)` of `np.ndarray` of shape `(n_samples,)`.

**Notes:**

- Computes `y_hat ± z_{(1+alpha)/2} · delta`.
- **No finite-sample guarantee.** Use `predict_interval_conformal` for guaranteed coverage.

**Example:**

```python
lo, hi = model.predict_interval(X_test, alpha=0.90)
```

---

### `fit_conformal(X_cal, y_cal, X_val, y_val, alpha=0.05, h_grid=None, lambda_penalty=10.0, verbose=False)`

Tune calibration bandwidth and store calibration residuals.

**Parameters:**

- `X_cal`, `y_cal` — calibration set, disjoint from training.
- `X_val`, `y_val` — validation set, disjoint from both training and calibration.
- `alpha` — float. Miscoverage level (e.g. `0.1` for 90% intervals).
- `h_grid` — optional array of candidate bandwidths. If `None`, uses `np.logspace(log10(0.5), log10(10), 20)`.
- `lambda_penalty` — unused (kept for backward compatibility). Coverage constraint is hard.
- `verbose` — if `True`, print tuned `h_cal` and validation coverage.

**Returns:** `self`.

**Notes:**

- Must be called **after** `fit`.
- Selects `h_cal` by grid search on validation, maximizing coverage subject to a 2% slack, then minimizing width.
- Stores residuals $R_j = |y_j^{\text{cal}} - \hat{y}(x_j^{\text{cal}})|$ in **original target units**.

**Example:**

```python
model.fit(X_train, y_train)
model.fit_conformal(
    X_cal, y_cal, X_val, y_val,
    alpha=0.1, verbose=True,
)
```

---

### `predict_interval_conformal(X, alpha=None)`

Return weighted conformal interval with finite-sample coverage guarantee.

**Parameters:**

- `X` — array-like of shape `(n_samples, n_features)`.
- `alpha` — miscoverage level. If `None`, uses the value from `fit_conformal`.

**Returns:** tuple `(lower, upper)` of `np.ndarray` of shape `(n_samples,)`.

**Notes:**

- Requires `fit_conformal` to have been called.
- Uses weighted quantile with kernel weights from `h_cal`.
- Coverage guarantee $\ge 1 - \alpha$ under exchangeability.
- Complexity: $O(q \cdot m \log m)$ for $q$ queries and $m$ calibration points.

**Example:**

```python
lo, hi = model.predict_interval_conformal(X_test, alpha=0.1)
coverage = np.mean((y_test >= lo) & (y_test <= hi))
```

---

### `get_h()`

Return the learned bandwidth vector.

**Returns:** `np.ndarray` of shape `(n_features,)`.

**Notes:**

- In **standardized** coordinates.
- Small values indicate highly relevant features; large values indicate features the model ignores.

**Example:**

```python
h = model.get_h()
print("Feature relevance:", 1.0 / h)
```

---

### `get_h_cal()`

Return the tuned calibration bandwidth vector.

**Returns:** `np.ndarray` of shape `(n_features,)`.

**Notes:**

- Available only after `fit_conformal`.
- May differ from `get_h()` — prediction and calibration have different optimal scales.

---

## Complete example

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from smallmlp import SmallMLPRegressor

# load and split
X, y = load_diabetes(return_X_y=True)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0)
X_cal, X_val, y_cal, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

# fit point predictor
model = SmallMLPRegressor(verbose=True)
model.fit(X_tr, y_tr)

# evaluate point prediction
y_val_hat = model.predict(X_val)
print(f"MAE: {mean_absolute_error(y_val, y_val_hat):.2f}")

# inspect learned bandwidth
h = model.get_h()
print(f"h: {h}")

# calibrate intervals
model.fit_conformal(X_cal, y_cal, X_val, y_val, alpha=0.1, verbose=True)

# evaluate coverage
lo, hi = model.predict_interval_conformal(X_val, alpha=0.1)
coverage = np.mean((y_val >= lo) & (y_val <= hi))
width = np.mean(hi - lo)
print(f"Coverage: {coverage:.3f}  (target ≥ 0.90)")
print(f"Mean width: {width:.2f}")
```

---

## Error handling

| Exception          | When                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------- |
| `ValueError`     | `X` has wrong number of features.                                                   |
| `NotFittedError` | `predict` / `predict_zone` called before `fit`.                                 |
| `NotFittedError` | `predict_interval_conformal` called before `fit_conformal`.                       |
| `ValueError`     | `fit_conformal` called with non-disjoint sets (not checked, but breaks guarantees). |

---

## Compatibility

- **scikit-learn API.** `SmallMLPRegressor` follows the `BaseEstimator` / `RegressorMixin` interface. It works with `clone`, `Pipeline`, `cross_val_score`, `GridSearchCV`.
- **GridSearchCV** will work but is unnecessary — the model has no critical hyperparameters to tune.
- **Pipeline** works, but external scaling is redundant (SmallMLP standardizes internally).

---

## Internal modules

| Module                 | Purpose                                                 |
| ---------------------- | ------------------------------------------------------- |
| `smallmlp.estimator` | `SmallMLPRegressor` class                             |
| `smallmlp.core`      | Forward pass: kernel-weighted mean + zone               |
| `smallmlp.kernels`   | Gaussian kernel, effective sample size, masking         |
| `smallmlp.loss`      | Huber loss, LOO-Huber loss                              |
| `smallmlp.conformal` | Weighted quantile, conformal intervals,`h_cal` tuning |

These are considered internal. The public API is `SmallMLPRegressor`.
