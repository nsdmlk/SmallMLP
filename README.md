# SmallMLP

> Adaptive neural networks for small nonlinear data, with calibrated conformal intervals.

[![PyPI version](https://img.shields.io/badge/pypi-v0.2.0-blue.svg)](https://pypi.org/project/smallmlp/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SmallMLP** is a library for **small, nonlinear datasets** ($n < 500$). It replaces manual architecture search with **adaptive hyperparameters derived from dataset parameters** ($n$, $d$, $K$), and produces **calibrated prediction intervals** through weighted conformal prediction.

- **Regression:** **25 / 45 wins**, best mean rank (1.82) among 5 models — no tuning.
- **Intervals:** **19% narrower** than split conformal at equal coverage (**31 / 32 wins**).
- **Classification:** best mean rank among 10 models on 18 datasets; significantly better than LogReg, KNN, SVC; statistically indistinguishable from MLP-128 and RF.

---

## Why SmallMLP?

Standard MLPs need architecture search. Kernel methods need bandwidth tuning. Bayesian methods scale poorly. SmallMLP removes all three problems:

| Property                                | Standard MLP | GP | SmallMLP |
| --------------------------------------- | ------------ | -- | -------- |
| Works on small data ($n < 500$)         | ✗            | ✓  | ✓        |
| No hyperparameter tuning                | ✗            | ✗  | ✓        |
| Prediction intervals / sets             | ✗            | ✓  | ✓        |
| Trains in seconds on $n < 500$          | ✓            | ✗  | ✓        |
| Multiclass-aware width                  | ✗            | —  | ✓        |

**Adaptive by construction.** Layer width, dropout, and weight decay are **computed from dataset parameters** — not searched.

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
model.fit(X_tr, y_tr)
model.fit_conformal(X_cal, y_cal, X_val, y_val, alpha=0.1)

y_hat = model.predict(X_val)
lo, hi = model.predict_interval_conformal(X_val, alpha=0.1)

coverage = np.mean((y_val >= lo) & (y_val <= hi))
print(f"Coverage: {coverage:.3f}  Width: {np.mean(hi - lo):.2f}")
```

---

## Quick start — classification

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from smallmlp import SmallMLPClassifier

X, y = load_iris(return_X_y=True)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
X_cal, X_val, y_cal, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=0)

clf = SmallMLPClassifier(class_weight=None)
clf.fit(X_tr, y_tr)
clf.fit_conformal(X_cal, y_cal, X_val, y_val, alpha=0.1)

probs = clf.predict_proba(X_val)
sets = clf.predict_set(X_val, alpha=0.1)

coverage = np.mean([y_val[i] in sets[i] for i in range(len(y_val))])
avg_size = np.mean([len(s) for s in sets])
print(f"Coverage: {coverage:.3f}  Avg set size: {avg_size:.3f}")
```

---

## Method

SmallMLP is an **adaptive MLP backbone** with **weighted conformal** uncertainty on top.

### 1. Adaptive width

Layer width is computed from dataset parameters. First layer:

$$
w_1 = \max\left(2K, \; \min\left(\left\lfloor \alpha \cdot \sqrt{n \cdot K \cdot \sqrt{d}} \right\rfloor, \; 4n\right)\right)
$$

Subsequent layers:

$$
w_l = \max\left(K, \; \left\lfloor w_1 \cdot \beta^{\,l-1} \right\rfloor\right)
$$

with $\alpha = 4.0$ and $\beta = 0.7$ chosen by empirical sweep.

**Properties:**

- Grows sublinearly with $n$, $K$ (as $\sqrt{\cdot}$), and $d$ (as $d^{1/4}$).
- First layer bounded below by $2K$; later layers by $K$.
- Capped at $4n$ to prevent blow-up on tiny datasets.
- Depth decay $\beta^{l-1}$ shrinks later layers.

For **regression**, we fall back to a fixed width $\min(2n, 128)$.

### 2. Adaptive dropout and weight decay

Weight decay is derived from $n$:

$$
\lambda = \frac{1}{n}
$$

Dropout uses a **clamped** function of $d/n$:

$$
p_{\text{drop}} = \min\left(\max\left(3 \cdot \frac{d}{n}, \; 0.1\right), \; 0.5\right)
$$

Raw $d/n$ systematically under-estimates required regularization on small data (sweep: optimum in $[0.1, 0.2]$, raw ratio gives $0.02$–$0.08$).

### 3. Training

Adam optimizer, early stopping on validation loss (20% of train). Class weights are applied only when the imbalance ratio exceeds $2{:}1$.

### 4. Weighted conformal intervals and sets

The backbone produces an embedding $h(x)$. Given calibration residuals $R_j = |y_j - \hat{y}(x_j)|$ (regression) or scores $s_j = 1 - \hat{p}_{y_j}(x_j)$ (classification), the interval or set for a new $x^*$ uses a **weighted quantile** of the calibration scores, with weights derived from the kernel:

$$
w_j(x^{\ast}) = \exp\left(-\frac{\|h(x^{\ast}) - h(x_j)\|^2}{2 h_{\text{cal}}^2}\right)
$$

This gives:

- **Finite-sample coverage guarantee** under exchangeability.
- **Locally adaptive** widths — narrow where data is dense, wide in empty regions.
- **Tuning-free calibration** — $h_{\text{cal}}$ selected by grid search on held-out validation.

---

## Results

### Regression — point prediction

45 small regression datasets ($n < 500$), 5-fold CV, mean absolute error.

| Model              | Mean rank ↓ | Mean MAE ↓ | Wins / 45 |
| ------------------ | ----------- | ---------- | --------- |
| **SmallMLP**       | **1.82**    | **8.43**   | **25**    |
| RF (100)           | 2.33        | 13.58      | 14        |
| MLP (256, 128)     | 3.22        | 11.57      | 4         |
| KNN ($k=5$)        | 3.67        | 18.89      | 1         |
| MLP (100,)         | 3.96        | 19.07      | 1         |

**SmallMLP beats MLP (256, 128) by 27% on MAE** without any tuning.

### Regression — prediction intervals

45 datasets, $\alpha = 0.1$ (target coverage ≥ 0.90), 60/20/20 split.

| Method                          | Valid (cov ≥ 0.88) | Mean width among valid ↓ |
| ------------------------------- | ------------------ | ------------------------ |
| **Weighted conformal**          | **38 / 45**        | **51.8**                 |
| Split conformal                 | 32 / 45            | 64.0                     |
| Heuristic $\hat{y} \pm z\sigma$ | 17 / 45            | 145.7                    |

Among 32 datasets where both weighted and split conformal are valid, **weighted conformal produces narrower intervals on 31** (mean width ratio **0.81**).

### Classification — full benchmark

18 small classification datasets, 5-fold CV, 10 models. Metric: **mean rank** (lower is better, rank 1 = best on that dataset).

| Model                | Mean rank ↓ | Mean acc ↑ | Wins / 18 |
| -------------------- | ----------- | ---------- | --------- |
| **SmallMLP (classic)** | **4.19**  | **0.8120** | **3**     |
| **SmallMLP (formula)** | **4.50**  | 0.8117     | 2         |
| MLP (128,)           | 4.64        | 0.8132     | 3         |
| MLP (100,)           | 5.25        | 0.8101     | 1         |
| SVC (RBF)            | 5.31        | 0.7990     | 2         |
| MLP (100, 100)       | 5.56        | 0.8037     | 3         |
| RF (100)             | 5.92        | 0.8035     | 6         |
| LogReg               | 6.17        | 0.7920     | 4         |
| KNN ($k=5$)          | 8.17        | 0.7715     | 2         |

**Wilcoxon signed-rank vs SmallMLP (formula), paired across 18 datasets:**

| Baseline     | p-value | Significant? |
| ------------ | ------- | ------------ |
| MLP (128,)   | 1.00    | no           |
| MLP (100,)   | 0.62    | no           |
| MLP (100,100) | 0.26   | no           |
| RF (100)     | 0.37    | no           |
| SmallMLP classic | 0.62 | no          |
| SVC (RBF)    | 0.047   | **yes** (SmallMLP better) |
| LogReg       | 0.013   | **yes** (SmallMLP better) |
| KNN ($k=5$)  | 0.002   | **yes** (SmallMLP better) |

**Summary:** SmallMLP achieves the best mean rank among 10 models, is **statistically indistinguishable** from tuned MLPs and RF, and **significantly better** than LogReg, KNN, and SVC.

### Classification — per-K-bucket accuracy

| Bucket                | SmallMLP (formula) | MLP (128,) | RF (100) |
| --------------------- | ------------------ | ---------- | -------- |
| Binary ($K=2$)        | 0.8675             | **0.8697** | 0.8521   |
| Multiclass ($2<K\le5$)| 0.8537             | **0.8531** | 0.8210   |
| Hard multiclass ($K>5$)| 0.6776            | 0.6798     | **0.7019** |

SmallMLP competitive on binary and moderate multiclass; RF dominates on hard multiclass.

### Classification — adaptive dropout ablation

9 small classification datasets, 10-fold CV, accuracy.

| Dropout mode                                      | Mean accuracy ↑ |
| ------------------------------------------------- | --------------- |
| Unclamped $\min(d/n, \; 0.5)$                     | 0.8214          |
| **Clamped $\min(\max(3d/n, \; 0.1), \; 0.5)$**    | **0.8269**      |
| Fixed $p = 0.2$                                   | 0.8278          |

Clamped dropout matches fixed $p = 0.2$ within noise; advantage concentrated on multiclass ($K \geq 6$) and high-dimensional ($d \geq 30$) datasets.

### Classification — conformal prediction sets

15 small datasets (binary and multiclass), 60/20/20 split, $\alpha = 0.1$.

| Dataset        | $K$ | Coverage | Avg set size |
| -------------- | --- | -------- | ------------ |
| iris           | 3   | 1.00     | 1.07         |
| wine           | 3   | 0.97     | 1.00         |
| breast cancer  | 2   | 0.97     | 1.00         |
| hepatitis      | 2   | 0.90     | 1.26         |
| parkinsons     | 2   | 1.00     | 1.05         |
| ionosphere     | 2   | 0.93     | 1.06         |
| banknote       | 2   | 0.99     | 1.00         |
| vowel ($K=26$) | 26  | 0.92     | 2.54         |

Coverage **≥ 0.90** on the majority of datasets; set size **close to 1** on easy problems, **grows** on hard multiclass ones.

---

## When to use SmallMLP

**Good fit:**

- Small datasets ($n < 500$) with nonlinear structure.
- Scientific instruments: multi-sensor arrays, medical cohorts, chemistry.
- When uncertainty quantification matters (intervals with coverage guarantee).
- When you don't want to tune architecture or regularization.

**Not a good fit:**

- Large datasets ($n > 10{,}000$) — more data favors standard MLPs.
- Linear problems — logistic regression or ridge is better.
- Very high dimensions ($d > 200$) with tiny $n$ — kernel methods suffer.

---

## Use cases

SmallMLP is designed for scientific instruments that produce many features per observation but few observations overall:

- **Telescopes with multi-sensor arrays:** 30 sensors × 100–200 observations per campaign.
- **Particle detectors:** 10–100 features per event, 200–1000 events per analysis.
- **Medical cohorts:** 10–50 clinical variables, 50–300 patients.
- **Chemistry / catalysis:** 10–40 descriptors, 50–200 experiments.

In all these settings, standard MLPs overfit, Gaussian Processes scale poorly, and hyperparameter tuning is impractical. SmallMLP provides adaptive regression and classification with calibrated conformal intervals — no tuning required.

---

## Limitations

- **$n < 500$.** Fit cost grows with $n$, width, and epochs.
- **Heteroscedastic residuals.** On ~15% of regression datasets, weighted conformal undercovers. Attributed to strong heteroscedasticity under non-exchangeability.
- **Linear problems.** SmallMLP loses to logistic regression and ridge on linear or near-linear tasks.
- **Raw classification accuracy.** SmallMLP matches but does not beat tuned MLPs and RF. Its advantages are **tuning-free operation**, **calibrated uncertainty**, and **competitive mean rank**.
- **Adaptive dropout is only marginally better than fixed $p = 0.2$ on average.** Benefit is regime-specific.

### Negative results

We document what we tried that did not work, to guide future work:

- **Adaptive depth** $L = f(n, d, K)$: no significant improvement over fixed $L = 2$ on $n < 500$.
- **PCA-based effective dimensionality** $d_{\text{eff}}$ in the width formula: no improvement over raw $d$.
- **Per-class sample count** $\log_2(n/K)$ and **softened depth decay** $\sqrt{d}/\sqrt{l}$ in the binary width formula: within noise.
- **Bias initialization sweep** (zeros, positive, uniform): none beat PyTorch Kaiming default. Correlation between bias negativity and accuracy was spurious.
- **Activation sweep** (tanh, GELU, SiLU, algebraic sigmoid, softsign): ReLU wins on 18 small classification datasets; smooth/saturating activations underperform.
- **Sqrt-balanced class weights**: improve over linear weights under extreme imbalance, but neither beats no weighting on accuracy.
- **Reweighting / soft-median** in conformal scores: does not improve coverage.
- **Heteroscedastic head:** degrades point predictions.

### What we tried that *did* work

- **Clamped adaptive dropout** `min(max(3d/n, 0.1), 0.5)` — better than raw `d/n`.
- **Sqrt-balanced class weights** — safer than linear under extreme imbalance (though not better than no weights for accuracy).
- **Weighted conformal** — 19% narrower intervals at equal coverage.
- **Multi-class-aware width** — competitive with fixed-width baselines without tuning.

---

## API overview

### `SmallMLPRegressor`

```python
SmallMLPRegressor(
    activation="relu",
    lr=1e-3,
    weight_decay=None,   # default 1/n
    max_epochs=500,
    patience=30,
    batch_size=None,     # default min(n, 32)
    val_frac=0.2,
    random_state=42,
    verbose=False,
)
```

**Methods:** `fit`, `predict`, `fit_conformal`, `predict_interval_conformal`.

### `SmallMLPClassifier`

```python
SmallMLPClassifier(
    activation="relu",          # 'relu' | 'tanh' | 'gelu' | 'silu' | 'algsig' | 'softsign'
    lr=1e-3,
    weight_decay=None,          # default 1/n
    max_epochs=500,
    patience=30,
    batch_size=None,
    val_frac=0.2,
    class_weight=None,          # None or "balanced"
    random_state=42,
    verbose=False,
    width_mode="formula",       # "formula" or "classic"
    alpha=4.0,                  # width scale
    beta=0.7,                   # depth decay
    bias_init="kaiming",        # "kaiming" | "zeros" | "positive" | "uniform"
)
```

**Methods:** `fit`, `predict`, `predict_proba`, `fit_conformal`, `predict_set`, `predict_set_labels`.

---

## License

MIT License. See `LICENSE` for details.

---

## Acknowledgments

Built independently during undergraduate studies at Beijing Institute of Technology. Inspired by the author's earlier work on robust gradient boosting for small data ([SmallGBM](https://github.com/nsdmlk/smallgbm)).


**Проверь, что в `Results` таблица согласована с реальными числами** — если после обновления кода числа изменились, обнови таблицу.
