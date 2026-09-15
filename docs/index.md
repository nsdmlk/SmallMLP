# SmallMLP

> Learned-bandwidth Nadaraya-Watson regression with prediction intervals for small nonlinear data.

**SmallMLP** is a non-parametric regressor designed for **small, nonlinear datasets** ($n < 500$). It learns a per-feature kernel bandwidth via leave-one-out optimization and produces **calibrated prediction intervals** through weighted conformal prediction.

On 45 benchmark datasets, SmallMLP outperforms standard MLPs without any hyperparameter tuning (25 / 45 wins, mean rank 1.80) and produces intervals **19% narrower** than split conformal at equal coverage (31 / 32 wins).

---

## Installation

```bash
pip install smallmlp
```

**Requirements:** Python 3.10+, `torch>=2.0`, `scikit-learn>=1.3`, `numpy`, `scipy`.

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
model.fit(X_tr, y_tr)                                     # point predictor
model.fit_conformal(X_cal, y_cal, X_val, y_val, alpha=0.1) # calibrate intervals

y_hat = model.predict(X_val)                              # point predictions
lo, hi = model.predict_interval_conformal(X_val, alpha=0.1)  # 90% intervals

coverage = np.mean((y_val >= lo) & (y_val <= hi))
print(f"Coverage: {coverage:.3f}")   # ~0.90
```

That's it. No hyperparameter tuning, no model selection, no feature scaling.

---

## When to use SmallMLP

**Good fit:**

- Small datasets ($n < 500$) with nonlinear structure.
- Scientific data — biology, medicine, physics, chemistry.
- Applications where prediction intervals matter.
- When you don't want to tune anything.

**Not a good fit:**

- Large datasets ($n > 10{,}000$) — kernel methods scale poorly.
- Linear problems — standard MLPs and linear models are better.
- Classification — SmallMLP is regression-only.

See [Limitations](limitations.md) for the full picture.

---

## Why SmallMLP?

|                                   | Standard MLP | Gaussian Process | **SmallMLP** |
| --------------------------------- | ------------ | ---------------- | ------------------ |
| Works on small data ($n < 500$) | ✗           | ✓               | ✓                 |
| Automatic bandwidth / no tuning   | ✗           | ✗               | ✓                 |
| Prediction intervals              | ✗           | ✓               | ✓                 |
| Non-parametric                    | ✗           | ✓               | ✓                 |
| Fast on$n < 500$                | ✓           | ✗               | ✓                 |

SmallMLP fills the gap: **non-parametric**, **tuning-free**, **fast**, **with calibrated uncertainty** — for small nonlinear regression.

---

## Method in one paragraph

SmallMLP is a Nadaraya-Watson kernel regressor with a **learned per-feature bandwidth** $h \in \mathbb{R}^d$, optimized by leave-one-out Huber loss via L-BFGS. Point predictions are kernel-weighted means. Prediction intervals come from **weighted conformal prediction**: calibration residuals are reweighted by the same kernel, and a finite-sample quantile gives **guaranteed coverage** under exchangeability — with intervals that adapt to local data density.

Full mathematics: [Method](method.md).

---

## Results at a glance

**Point prediction** (45 datasets, 5-fold CV, MAE):

| Model              | Mean rank ↓   | Wins / 45    |
| ------------------ | -------------- | ------------ |
| **SmallMLP** | **1.80** | **25** |
| MLP (256, 128)     | 2.76           | 11           |
| KNN (k=10)         | 3.36           | 5            |
| MLP (100,)         | 4.33           | 0            |

**Prediction intervals** (45 datasets, $\alpha = 0.1$, target coverage ≥ 0.90):

| Method                                  | Valid datasets    | Mean width ↓  |
| --------------------------------------- | ----------------- | -------------- |
| **Weighted conformal (SmallMLP)** | **38 / 45** | **51.8** |
| Split conformal                         | 32 / 45           | 64.0           |
| Heuristic zone                          | 17 / 45           | 145.7          |

Among 32 datasets where both methods are valid, weighted conformal produces **narrower intervals on 31**.

Full results: [Experiments](experiments.md).

---

## Documentation

| Document                     | Contents                                                    |
| ---------------------------- | ----------------------------------------------------------- |
| [Method](method.md)           | Full mathematics: point predictor, zone, weighted conformal |
| [API](api.md)                 | Complete API reference for`SmallMLPRegressor`             |
| [Experiments](experiments.md) | Benchmark setup, datasets, results                          |
| [Limitations](limitations.md) | Negative results, applicability domain                      |
| [Citation](citation.md)       | How to cite SmallMLP                                        |

---

## Citation

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

A preprint is in preparation.

---

## License

MIT. See [LICENSE](https://github.com/nsdmlk/smallmlp/blob/main/LICENSE).
