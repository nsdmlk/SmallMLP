# SmallMLP

> Non-parametric regression and classification for small nonlinear data, with calibrated prediction intervals.

[![PyPI version](https://img.shields.io/badge/pypi-v0.2.0-blue.svg)](https://pypi.org/project/smallmlp/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](<https://img.shields.io/badge/tests-3%20passed-brightgreen.svg>)](#testing)
[![arXiv](<https://img.shields.io/badge/arXiv-coming%20soon-red.svg>)](#citation)

**SmallMLP** is a non-parametric library for **small, nonlinear datasets** ($n < 500$). It learns a per-feature kernel bandwidth via leave-one-out optimization and produces **calibrated prediction intervals** through weighted conformal prediction.

- **Regression:** 25 / 45 wins against standard MLPs, no hyperparameter tuning.
- **Classification:** competitive on nonlinear boundaries; 5 / 12 wins on synthetic nonlinear datasets.
- **Intervals:** 19% narrower than split conformal at equal coverage (31 / 32 wins).

---

## Why SmallMLP?

Standard MLPs overfit on small data. Kernel methods are robust but require manual bandwidth selection. Bayesian methods (GP, BNN) provide uncertainty but scale poorly. SmallMLP fills the gap:

| Property                               | Standard MLP | GP | SmallMLP |
| -------------------------------------- | ------------ | -- | -------- |
| Works on small data (n < 500)          | ✗           | ✓ | ✓       |
| Automatic bandwidth / no tuning        | ✗           | ✗ | ✓       |
| Prediction intervals                   | ✗           | ✓ | ✓       |
| Non-parametric (no fixed architecture) | ✗           | ✓ | ✓       |
| Trains in seconds on n < 500           | ✓           | ✗ | ✓       |

---

## Installation

```bash
pip install smallmlp
```

Requires Python 3.10+, `torch>=2.0`, `scikit-learn>=1.3`.

---

## Quick start — regression

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
model.fit(X_tr, y_tr)                                       # point predictor
model.fit_conformal(X_cal, y_cal, X_val, y_val, alpha=0.1) # calibrate intervals

y_hat = model.predict(X_val)
lo, hi = model.predict_interval_conformal(X_val, alpha=0.1)

coverage = np.mean((y_val >= lo) & (y_val <= hi))  # ~0.90
print(f"Coverage: {coverage:.3f}")
```

---

## Quick start — classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from smallmlp import SmallMLPClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0)
X_cal, X_val, y_cal, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

clf = SmallMLPClassifier(point_method="klr", class_weight="balanced")
clf.fit(X_tr, y_tr)
clf.fit_conformal(X_cal, y_cal, X_val, y_val, alpha=0.1)

p = clf.predict_proba(X_val)[:, 1]
sets = clf.predict_set(X_val, alpha=0.1)

coverage = np.mean([y_val[i] in sets[i] for i in range(len(y_val))])
avg_size = np.mean([len(s) for s in sets])
print(f"Coverage: {coverage:.3f}  Avg set size: {avg_size:.3f}")
```

---

## Method

SmallMLP combines two ideas.

**1. Learned-bandwidth Nadaraya-Watson point predictor.**
For regression:

$$
\hat{y}(x) = \frac{\sum_i w_i(x) \, y_i}{\sum_i w_i(x)}, \quad w_i(x) = \exp\left(-\frac{\|x - x_i\|^2}{2 h^2}\right)
$$

For classification, two point methods are available:

- `point_method="nw"` — kernel-weighted vote (same form, $y_i \in \{0, 1\}$).
- `point_method="klr"` — kernel logistic regression, $\hat{p}(x) = \sigma(\sum_i \alpha_i K(x, x_i) + b)$, with $\alpha, b, h$ learned jointly.

The bandwidth vector $h \in \mathbb{R}^d$ is **learned** by leave-one-out optimization (Huber for regression, BCE for classification). No manual tuning, no grid search. Each feature gets its own bandwidth, giving automatic relevance weighting.

**2. Weighted conformal prediction sets.**
Given calibration residuals $R_j$ (regression) or scores $s_j = 1 - \hat{p}_{y_j}(x_j)$ (classification), the interval or set for a new $x^*$ uses a **weighted quantile** of $\{R_j\}$ or $\{s_j\}$ with weights derived from the same kernel. This gives **finite-sample coverage guarantee** under exchangeability, with intervals and sets that **adapt to local data density** — narrow where data is dense, wide in empty regions.

---

## Results

### Regression — point prediction

45 small regression datasets (n < 500), 5-fold CV, mean absolute error.

| Model              | Mean rank ↓   | Mean MAE ↓     | Wins / 45    |
| ------------------ | -------------- | --------------- | ------------ |
| **SmallMLP** | **1.80** | **11.39** | **25** |
| MLP (256, 128)     | 2.76           | 12.44           | 11           |
| KNN (k=10)         | 3.36           | 19.13           | 5            |
| KNN (k=5)          | 3.49           | 18.89           | 2            |
| MLP (100,)         | 4.33           | 34.44           | 0            |
| MLP (32,)          | 5.27           | 51.95           | 0            |

### Regression — prediction intervals

45 datasets, $\alpha = 0.1$ (target coverage ≥ 0.90), 60/20/20 split.

| Method                                  | Valid (cov ≥ 0.88) | Mean width among valid ↓ |
| --------------------------------------- | ------------------- | ------------------------- |
| **Weighted conformal (SmallMLP)** | **38 / 45**   | **51.8**            |
| Split conformal                         | 32 / 45             | 64.0                      |
| Heuristic zone                          | 17 / 45             | 145.7                     |

Among 32 datasets where both weighted and split conformal are valid, **weighted conformal produces narrower intervals on 31** (mean width ratio 0.81).

### Classification — nonlinear synthetic

12 synthetic nonlinear datasets (moons, circles, XOR, spirals; n ∈ {100, 200, 400}), 5-fold CV, AUC.

| Model                   | Mean AUC ↑     | Mean rank ↓   | Wins / 12   |
| ----------------------- | --------------- | -------------- | ----------- |
| **SmallMLP (nw)** | **0.937** | **2.79** | **5** |
| MLP (100,)              | 0.932           | 3.42           | 0           |
| KNN (k=5)               | 0.927           | 4.75           | 2           |
| RF (100)                | 0.924           | 5.08           | 0           |
| MLP (32,)               | 0.903           | 3.75           | 3           |
| SmallMLP (klr)          | 0.863           | 4.67           | 2           |
| LogReg                  | 0.641           | 7.50           | 0           |

### Ablation — learned bandwidth matters

| Variant             | Mean rank ↓   | Wins / 18    |
| ------------------- | -------------- | ------------ |
| **Learned h** | **1.44** | **13** |
| Fixed h = 1.0       | 2.50           | 4            |
| Fixed h = 0.5       | 3.39           | 0            |
| Fixed h = 2.0       | 3.17           | 1            |
| Fixed h = 5.0       | 4.50           | 0            |

Learning the bandwidth is the core mechanism — fixed bandwidth loses most of the advantage.

---

## When to use SmallMLP

**Good fit:**

- Small datasets ($n < 500$) with nonlinear structure.
- Scientific instruments: telescopes, particle detectors, medical cohorts, chemistry.
- When uncertainty quantification matters (prediction intervals with coverage guarantee).
- When you don't want to tune hyperparameters.

**Not a good fit:**

- Large datasets ($n > 10{,}000$) — kernel methods scale poorly.
- Linear problems — standard MLPs and linear models are better.
- Classification on linearly separable data — use logistic regression or a standard MLP.

---

## Use cases

SmallMLP is designed for scientific instruments that produce many features per observation but few observations overall:

- **Telescopes with multi-sensor arrays:** 30 sensors × 100–200 observations per campaign.
- **Particle detectors:** 10–100 features per event, 200–1000 events per analysis.
- **Medical cohorts:** 10–50 clinical variables, 50–300 patients.
- **Chemistry / catalysis:** 10–40 descriptors, 50–200 experiments.

In all these settings, standard MLPs overfit, Gaussian Processes scale poorly, and hyperparameter tuning is impractical. SmallMLP provides non-parametric regression and classification with calibrated prediction intervals — no tuning required.

---

## Limitations

- **n < 500.** Inference cost scales linearly with training set size (O(n) per prediction). Fit cost is O(n²).
- **Heteroscedastic residuals.** On 7 / 45 regression datasets, weighted conformal undercovers. We attribute this to strongly heteroscedastic noise, a known limitation of weighted conformal under non-exchangeability.
- **Linear problems.** SmallMLP loses to standard MLPs and logistic regression on linear or near-linear tasks. This is expected for a non-parametric method.
- **Classification on UCI datasets.** On 18 small binary UCI datasets, SmallMLP did not consistently outperform standard classifiers. The advantage appears on genuinely nonlinear boundaries (moons, spirals), not on linearly separable data.

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

**Methods:** `fit`, `predict`, `predict_zone`, `predict_interval`, `fit_conformal`, `predict_interval_conformal`, `get_h`.

### `SmallMLPClassifier`

```python
SmallMLPClassifier(
    point_method="nw",    # "nw" (kernel vote) or "klr" (kernel logistic regression)
    h_min=0.01,
    h_max=10.0,
    lam=1e-3,             # L2 strength (klr only)
    class_weight=None,    # None or "balanced"
    max_iter=30,
    inner_iter=10,
    verbose=False,
)
```

**Methods:** `fit`, `predict`, `predict_proba`, `fit_conformal`, `predict_set`, `get_h`.

---

## Testing

```bash
pytest tests/ -v
```

Three synthetic tests verify core regression properties:

- Zone grows in empty regions.
- Zone grows around outliers.
- Point prediction is robust to outliers.

---

## Citation

If you use SmallMLP in your research, please cite:

```bibtex
@software{emelyanov2026smallmlp,
  author    = {Emelyanov, Ilya},
  title     = {{SmallMLP}: Non-parametric regression and classification
               for small nonlinear data with conformal prediction intervals},
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
