
# Limitations

SmallMLP is not a universal regressor. This document describes where it works, where it does not, and why.

We include **negative results** explicitly — methods that were tested and rejected, and failure modes observed in benchmarks. This is intended to save time for future researchers and to set honest expectations for users.

---

## 1. Applicability domain

SmallMLP is designed for a specific regime:

| Condition      | Value                                  |
| -------------- | -------------------------------------- |
| Sample size    | $n < 500$                            |
| Relationship   | Nonlinear                              |
| Noise          | Moderate, not strongly heteroscedastic |
| Dimensionality | $d$ small to moderate ($< 100$)    |

Outside this regime, other methods are likely better.

---

## 2. What SmallMLP does not do well

### 2.1 Linear or near-linear problems

On linear problems, SmallMLP **loses to standard MLPs and linear models**.

Examples from benchmarks:

| Dataset          | SmallMLP MAE | Best competitor | Competitor |
| ---------------- | ------------ | --------------- | ---------- |
| diabetes         | 47.5         | 45.6            | MLP_wide   |
| diabetes_100     | 48.1         | 40.2            | MLP_wide   |
| mreg_100x20_inf5 | 26.9         | 22.4            | MLP_wide   |
| friedman3_300    | 0.114        | 0.110           | MLP_wide   |

**Why.** Kernel smoothing is local. It cannot extrapolate a global linear trend beyond the training data range, and it cannot learn a linear combination of features as efficiently as a parametric model. On linear tasks, the softness of the kernel-weighted mean becomes a liability.

**Recommendation.** For linear problems, use ridge regression, `sklearn.linear_model`, or a standard MLP. SmallMLP is not the right tool.

### 2.2 Large datasets

Complexity:

| Operation                      | Complexity                                |
| ------------------------------ | ----------------------------------------- |
| Fit                            | $O(n^2 d)$ per L-BFGS forward           |
| Predict (per query)            | $O(n \cdot d)$                          |
| Conformal interval (per query) | $O(n_{\text{cal}} \log n_{\text{cal}})$ |

For $n = 10{,}000$, fit time exceeds 100 seconds, and prediction memory is ~800 MB for the kernel matrix. **SmallMLP is not designed for large data.**

**Recommendation.** For $n > 10{,}000$, use gradient boosting, random forests, or deep MLPs.

### 2.3 Classification

`SmallMLPClassifier` was implemented and tested on 18 small binary classification datasets. Results:

| Model               | Mean AUC | Mean rank ↓   |
| ------------------- | -------- | -------------- |
| Random Forest (100) | 0.940    | 3.38           |
| MLP (32,)           | 0.930    | 3.58           |
| MLP (100,)          | 0.930    | 3.67           |
| **SmallMLP**  | 0.922    | **5.58** |
| KNN (k=10)          | 0.918    | 5.54           |
| Logistic Regression | 0.910    | 4.13           |

SmallMLP ranked **5.6 / 8** — behind all MLP variants and Random Forest.

**Why.** Soft-median / soft-vote in $\{0,1\}$ space loses information about class confidence. Parametric and ensemble methods exploit correlated features better. Weighted kernel methods do not have an advantage in classification with limited data.

**Recommendation.** For binary classification, use logistic regression, gradient boosting, or a standard MLP.

---

## 3. Negative results (methods tested and rejected)

These methods were part of earlier SmallMLP versions and were removed after empirical evaluation.

### 3.1 Soft-median via iterative reweighting

An early version used iteratively reweighted soft-median (IRLS with a Gaussian reweighting kernel):

$$
\tilde{w}_i \propto w_i \cdot \exp(-|y_i - \hat{y}| / \tau)
$$

Four ablations on 45 datasets:

| Variant                               | Mean rank ↓   |
| ------------------------------------- | -------------- |
| τ = 10 (reweighting effectively off) | **2.31** |
| τ = 1 (constant)                     | 2.36           |
| τ = 0.1                              | 2.60           |
| Auto τ (original formula)            | 3.18           |
| IRLS 5 iterations                     | 4.42           |
| IRLS 10 iterations                    | 4.76           |

**All reweighting variants performed worse than no reweighting.** The best configuration was one where reweighting was effectively disabled. Removed entirely.

**Why.** Soft-median converges to the median, which discards information about residual magnitude. On small data, every point carries information; robust reweighting is counterproductive.

### 3.2 Automatic temperature

The original formula for reweighting temperature:

$$
\tau(x) = \delta_0(x) / \sqrt{n_{\text{eff}}(x)}
$$

was tested against four alternative auto-forms and fixed values. All auto-forms were **worse** than a large constant $\tau = 10$ (i.e., reweighting off).

**Why.** The formula was derived from intuition about local density, but the empirical optimum is "no reweighting at all." Removed.

### 3.3 Automatic bandwidth selection via kernel density

We explored using the effective sample size $n_{\text{eff}}(x)$ to modulate the reweighting. This produced unstable results — the optimal $n_{\text{eff}}$-based scheme was dataset-dependent and did not generalize.

---

## 4. Failure modes observed

### 4.1 Undercowering on heteroscedastic residuals

On **7 of 45 datasets**, weighted conformal coverage fell below 0.88 (2% below the 0.90 target):

| Dataset           | Coverage (WC) | n_cal |
| ----------------- | ------------- | ----- |
| synth_poly_300x15 | 0.550         | 60    |
| friedman1_300     | 0.533         | 60    |
| friedman2_500     | 0.750         | 100   |
| friedman3_100     | 0.700         | 20    |
| diabetes_100      | 0.600         | 20    |
| wine_quality_100  | 0.650         | 20    |
| abalone_500       | 0.810         | 100   |

**Pattern:** all invalid datasets have **heteroscedastic residuals** or **heavy-tailed distributions**. On `synth_poly_300` and `friedman1_300`, coverage drops to 0.55 — a severe miss.

**Why.** Weighted conformal assumes **exchangeability** between calibration and test residuals. When residuals are strongly heteroscedastic (variance depends on $x$), the weighted quantile on calibration does not transfer to test.

**Mitigation.** None included in SmallMLP. Standard conformal theory offers corrections (e.g., Mondrian conformal, bootstrapped conformal) but they require additional assumptions. We document the failure and leave the fix to future work.

### 4.2 Undercowering on small calibration sets

For $n_{\text{cal}} < 20$, finite-sample coverage guarantee becomes weak. With $n_{\text{cal}} = 10$ (`diabetes_50`), coverage was 1.000 (overcoverage), but with $n_{\text{cal}} = 20$ (`diabetes_100`), coverage fell to 0.600.

**Recommendation.** Use at least 20% of data for calibration, and at least $n_{\text{cal}} \ge 30$ if possible.

### 4.3 Feature relevance collapse

On linear problems, some features' bandwidth may saturate at $h_{\max}$ — the model **ignores** those features. Example on diabetes:

h = [0.90, 1.13, 0.65, 1.21, 9.83, 0.91, 1.17, 9.91, 0.62, 9.90]

Features 4, 7, 9 are effectively dropped. This is **correct behavior** for a kernel method — those features contribute little locally — but it means SmallMLP **does not use all information** that a linear model would.

---

## 5. Comparison to alternatives

| Method             | Small data    | Nonlinear | Uncertainty | Tuning-free | Fast         |
| ------------------ | ------------- | --------- | ----------- | ----------- | ------------ |
| **SmallMLP** | ✓            | ✓        | ✓          | ✓          | ✓ (n < 500) |
| MLP                | ✗ (overfits) | ✓        | ✗          | ✗          | ✓           |
| Gaussian Process   | ✓            | ✓        | ✓          | ✗          | ✗ (O(n³))  |
| Gradient Boosting  | ✗ (overfits) | ✓        | ✗          | ✗          | ✓           |
| KNN                | ✓            | ✓        | ✗          | ✓          | ✓           |
| Linear regression  | ✓            | ✗        | ✓          | ✓          | ✓           |

**SmallMLP is not the best at any single criterion** — GP has better uncertainty, MLP is more flexible, linear regression is faster. It occupies a **specific niche**: tuning-free, non-parametric, calibrated intervals, small nonlinear data.

---

## 6. What would improve SmallMLP

Open directions (not implemented):

1. **Mondrian conformal** — partition calibration by feature values to handle heteroscedasticity.
2. **Bootstrapped conformal** — resample calibration for tighter finite-sample bounds.
3. **Sparse kernel** — restrict to $k$-nearest neighbors to reduce $O(n^2)$ to $O(nk)$.
4. **Inductive conformal for streaming** — for online updates.
5. **Multi-output regression** — currently single-target only.

---

## 7. Honest summary

**SmallMLP works when:**

- Data is small ($n < 500$).
- The relationship is nonlinear.
- You need calibrated intervals.
- You don't want to tune hyperparameters.

**SmallMLP does not work when:**

- Data is large.
- The problem is linear.
- The task is classification.
- Residuals are strongly heteroscedastic (7 / 45 datasets).

**SmallMLP is not a replacement for GP, MLP, or gradient boosting.** It is a specific tool for a specific regime. Use it where it fits.
