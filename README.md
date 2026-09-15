# SmallMLP

> Learned-bandwidth Nadaraya-Watson regression with prediction intervals for small nonlinear data.

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/smallmlp/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-3%20passed-brightgreen.svg)](#testing)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-red.svg)](#citation)

**SmallMLP** is a non-parametric regressor designed for **small, nonlinear datasets** (n < 500). It learns a per-feature kernel bandwidth via leave-one-out optimization and produces **calibrated prediction intervals** through weighted conformal prediction. On 45 benchmark datasets it outperforms standard MLPs without any hyperparameter tuning, and produces intervals **19% narrower** than split conformal at equal coverage.

---

## Why SmallMLP?

Standard MLPs overfit on small data. Kernel methods are robust but require manual bandwidth selection. Bayesian methods (GP, BNN) provide uncertainty but scale poorly. SmallMLP fills the gap:

| Property | Standard MLP | GP | SmallMLP |
|---|---|---|---|
| Works on small data (n < 500) | ✗ | ✓ | ✓ |
| Automatic bandwidth / no tuning | ✗ | ✗ | ✓ |
| Prediction intervals | ✗ | ✓ | ✓ |
| Non-parametric (no fixed architecture) | ✗ | ✓ | ✓ |
| Trains in seconds on n < 500 | ✓ | ✗ | ✓ |

---

## Installation

```bash
pip install smallmlp
```

Requires Python 3.10+, `torch>=2.0`, `scikit-learn>=1.3`.

---

## Quick start

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from smallmlp import SmallMLPRegressor

X, y = load_diabetes(return_X_y=True)

# 60 / 20 / 20 split: train / calibration / validation
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0)
X_cal, X_val, y_cal, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

model = SmallMLPRegressor()
model.fit(X_tr, y_tr)                          # point predictor
model.fit_conformal(X_cal, y_cal, X_val, y_val, alpha=0.1)  # calibrate intervals

y_hat = model.predict(X_val)                   # point predictions
lo, hi = model.predict_interval_conformal(X_val, alpha=0.1)  # 90% intervals

coverage = np.mean((y_val >= lo) & (y_val <= hi))  # ~0.90
print(f"Coverage: {coverage:.3f}")
```

---

## Method

SmallMLP combines two ideas:

**1. Learned-bandwidth Nadaraya-Watson regression.**
The point predictor is a kernel-weighted mean:
$$\hat{y}(x) = \frac{\sum_i w_i(x) \, y_i}{\sum_i w_i(x)}, \quad w_i(x) = \exp\left(-\frac{\|x - x_i\|^2}{2 h^2}\right)$$

The bandwidth vector $h \in \mathbb{R}^d$ is **learned** by minimizing a leave-one-out Huber loss — no manual tuning, no grid search. Each feature gets its own bandwidth, giving automatic relevance weighting.

**2. Weighted conformal prediction intervals.**
Given calibration residuals $R_j = |y_j - \hat{y}(x_j)|$, the interval for a new point $x^*$ is:
$$[\hat{y}(x^*) - \hat{q}(x^*), \; \hat{y}(x^*) + \hat{q}(x^*)]$$

where $\hat{q}(x^*)$ is a **weighted quantile** of $\{R_j\}$ with weights derived from the same kernel. This gives **finite-sample coverage guarantee** under exchangeability, with intervals that **adapt to local data density** — narrow where data is dense, wide in empty regions.

---

## Results

### Point prediction

45 small regression datasets (n < 500), 5-fold CV, mean absolute error.

| Model | Mean rank ↓ | Mean MAE ↓ | Wins / 45 |
|---|---|---|---|
| **SmallMLP** | **1.80** | **11.39** | **25** |
| MLP (256, 128) | 2.76 | 12.44 | 11 |
| KNN (k=10) | 3.36 | 19.13 | 5 |
| KNN (k=5) | 3.49 | 18.89 | 2 |
| MLP (100,) | 4.33 | 34.44 | 0 |
| MLP (32,) | 5.27 | 51.95 | 0 |

### Prediction intervals

45 datasets, α = 0.1 (target coverage ≥ 0.90), 60/20/20 split.

| Method | Valid (cov ≥ 0.88) | Mean width among valid ↓ |
|---|---|---|
| **Weighted conformal (SmallMLP)** | **38 / 45** | **51.8** |
| Split conformal | 32 / 45 | 64.0 |
| Heuristic zone | 17 / 45 | 145.7 |

Among 32 datasets where both weighted and split conformal are valid, **weighted conformal produces narrower intervals on 31** (mean width ratio 0.81).

### Ablation: learned bandwidth matters

| Variant | Mean rank ↓ | Wins / 18 |
|---|---|---|
| **Learned h (SmallMLP)** | **1.44** | **13** |
| Fixed h = 1.0 | 2.50 | 4 |
| Fixed h = 0.5 | 3.39 | 0 |
| Fixed h = 2.0 | 3.17 | 1 |
| Fixed h = 5.0 | 4.50 | 0 |

Learning the bandwidth is the core mechanism — fixed bandwidth loses most of the advantage.

---

## When to use SmallMLP

**Good fit:**
- Small datasets (n < 500) with nonlinear structure.
- Scientific data: biology, medicine, physics, chemistry.
- When uncertainty quantification matters (prediction intervals with coverage guarantee).
- When you don't want to tune hyperparameters.

**Not a good fit:**
- Large datasets (n > 10,000) — kernel methods scale poorly.
- Linear problems — standard MLPs and linear models are better.
- Classification — SmallMLP is designed for regression (see Limitations).

---

## Limitations

- **n < 500.** Inference cost scales linearly with training set size (O(n) per prediction). Fit cost is O(n²).
- **Regression only.** Classification benchmarks showed no consistent advantage over standard classifiers.
- **Heteroscedastic residuals.** On 7 / 45 datasets, weighted conformal undercovers. We attribute this to strongly heteroscedastic noise, a known limitation of weighted conformal under non-exchangeability.
- **Linear problems.** SmallMLP loses to standard MLPs on linear/near-linear tasks (e.g. diabetes, make_regression with informative subset). This is expected for a non-parametric method.

---

## API overview

### `SmallMLPRegressor`

```python
SmallMLPRegressor(
    h_min=0.01,           # lower bound for bandwidth
    h_max=10.0,           # upper bound
    max_iter=30,          # L-BFGS outer iterations
    inner_iter=10,        # L-BFGS inner iterations
    alpha=1e-3,           # prior strength for zone
    verbose=False,
)
```

**Methods:**
- `fit(X, y)` — train point predictor.
- `predict(X)` — point predictions.
- `predict_zone(X)` — heuristic zone (ŷ, δ) from local weighted variance.
- `predict_interval(X, alpha)` — Gaussian-style interval [ŷ ± z·δ].
- `fit_conformal(X_cal, y_cal, X_val, y_val, alpha)` — calibrate weighted conformal.
- `predict_interval_conformal(X, alpha)` — finite-sample conformal interval.
- `get_h()` — learned bandwidth vector.

---

## Testing

```bash
pytest tests/ -v
```

Three synthetic tests verify core properties:
- Zone grows in empty regions.
- Zone grows around outliers.
- Point prediction is robust to outliers.

---

## Citation

If you use SmallMLP in your research, please cite:

```bibtex
@software{emelyanov2026smallmlp,
  author    = {Emelyanov, Ilya},
  title     = {{SmallMLP}: Learned-bandwidth Nadaraya-Watson regression
               with prediction intervals for small nonlinear data},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/nsdmlk/smallmlp}
}
```

A preprint is in preparation. Check back for the arXiv link.

---

## Contributing

Issues and pull requests are welcome. See `CONTRIBUTING.md` for development setup.

---

## License

MIT License. See `LICENSE` for details.

---

## Acknowledgments

Built independently during undergraduate studies at Beijing Institute of Technology. Inspired by the author's earlier work on robust gradient boosting for small data ([SmallGBM](https://github.com/nsdmlk/smallgbm)).
