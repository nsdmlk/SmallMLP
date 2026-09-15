
# Method

This document describes the mathematics behind SmallMLP: the point predictor, the heuristic zone, and the weighted conformal intervals.

---

## 1. Problem setup

We consider regression on a small dataset $\{(x_i, y_i)\}_{i=1}^{n}$ with $n < 500$, $x_i \in \mathbb{R}^d$, $y_i \in \mathbb{R}$. The regime of interest is **small, nonlinear, heteroscedastic**: $n$ is too small for deep models to generalize, but the relationship between $x$ and $y$ is not linear.

Two problems arise:

1. **Point prediction.** A standard MLP with $p \gg n$ parameters has infinitely many zero-training-error solutions. Regularization picks one arbitrarily.
2. **Uncertainty.** Even a good point predictor gives no information about *where* it should be trusted. For scientific applications with high cost of error, this is unacceptable.

SmallMLP addresses both: a non-parametric point predictor with **learned bandwidth**, and **weighted conformal** intervals with finite-sample coverage.

---

## 2. Point predictor: learned-bandwidth Nadaraya-Watson

### 2.1 Kernel-weighted mean

The point predictor is a Nadaraya-Watson estimator:

$$
\hat{y}(x) = \frac{\sum_{i=1}^{n} w_i(x) \, y_i}{\sum_{i=1}^{n} w_i(x)}
$$

with Gaussian kernel weights:

$$
w_i(x) = \exp\left(-\frac{1}{2} \sum_{k=1}^{d} \frac{(x_k - x_{i,k})^2}{h_k^2}\right)
$$

The bandwidth $h \in \mathbb{R}_{>0}^d$ is a **vector**, one per feature. This gives automatic relevance weighting: features with small $h_k$ contribute strongly, features with large $h_k$ are effectively ignored.

### 2.2 Learning $h$ via leave-one-out

We learn $h$ by minimizing a leave-one-out Huber loss on the training set:

$$
\mathcal{L}(h) = \frac{1}{n} \sum_{j=1}^{n} \rho\!\left(y_j - \hat{y}^{(-j)}(x_j; h)\right)
$$

where $\hat{y}^{(-j)}$ is the prediction at $x_j$ **excluding** point $j$ from the kernel sum. This is implemented by masking the diagonal of the weight matrix $W \in \mathbb{R}^{n \times n}$.

The Huber loss with threshold $\delta = 1$ (after standardizing $y$):

$$
\rho(e) = \begin{cases} \tfrac{1}{2} e^2 & |e| \le 1 \\ |e| - \tfrac{1}{2} & |e| > 1 \end{cases}
$$

is robust to outliers and differentiable everywhere.

### 2.3 Bounded parametrization

To keep $h$ in a reasonable range, we parametrize it as:

$$
h_k = h_{\min} + (h_{\max} - h_{\min}) \cdot \sigma(\psi_k)
$$

where $\sigma$ is the sigmoid function and $\psi \in \mathbb{R}^d$ is the unconstrained parameter. Defaults: $h_{\min} = 0.01$, $h_{\max} = 10$. This prevents $h_k \to \infty$ (which would make the model globally constant) and $h_k \to 0$ (which would make it interpolate noise).

### 2.4 Optimization

We minimize $\mathcal{L}(h)$ over $\psi$ using L-BFGS with strong Wolfe line search. Because the entire forward pass is differentiable in $\psi$ (softplus, sigmoid, exp, division), gradients flow via autograd. The optimization converges in 3–5 outer iterations in practice.

---

## 3. Heuristic prediction zone

For a query point $x$, the **distance-aware zone** is:

$$
\delta(x) = \sqrt{\frac{\sum_i w_i(x) (y_i - \hat{y}(x))^2 + \alpha \, p(x) \, \sigma_y^2}{\sum_i w_i(x) + \alpha \, p(x)}}
$$

where:

- $\sigma_y^2$ is the global variance of $y$ (computed once on the training set).
- $p(x) = \mathrm{clamp}\left(1 - n_{\mathrm{eff}}(x) / n, \; 0, \; 1\right)$ is a measure of local emptiness.
- $n_{\mathrm{eff}}(x) = (\sum_i w_i)^2 / \sum_i w_i^2$ is the effective sample size around $x$.
- $\alpha = 10^{-3}$ is a small prior strength.

**Interpretation.** When $x$ is in a dense region, $p(x) \approx 0$ and $\delta(x)$ is the local weighted standard deviation. When $x$ is far from all training points, $p(x) \to 1$ and $\delta(x) \to \sigma_y$ — the model falls back to the global uncertainty.

**Property.** $\lim_{\|x\| \to \infty} \delta(x) = \sigma_y$. Far from data, the zone width equals the global standard deviation.

The heuristic interval is:

$$
[\hat{y}(x) - z_{1-\alpha/2} \cdot \delta(x), \; \hat{y}(x) + z_{1-\alpha/2} \cdot \delta(x)]
$$

where $z_{1-\alpha/2}$ is the Gaussian quantile. This interval has **no coverage guarantee** — it is a heuristic. The next section provides guaranteed intervals.

---

## 4. Weighted conformal intervals

### 4.1 Split conformal baseline

Given a calibration set $\{(x_j^{\mathrm{cal}}, y_j^{\mathrm{cal}})\}_{j=1}^{m}$ disjoint from training, compute residuals:

$$
R_j = |y_j^{\mathrm{cal}} - \hat{y}(x_j^{\mathrm{cal}})|
$$

Classical split conformal uses a **global** quantile:

$$
\hat{q} = Q_{\lceil (1-\alpha)(m+1) \rceil / m}\!\left(\{R_j\}\right)
$$

and the interval $[\hat{y}(x) - \hat{q}, \hat{y}(x) + \hat{q}]$ has finite-sample coverage guarantee $\ge 1 - \alpha$ under exchangeability. The drawback: **one width for all $x$**.

### 4.2 Weighted conformal

SmallMLP replaces the global quantile with a **weighted** one:

$$
\hat{q}(x) = \mathrm{WeightedQuantile}_{\lceil (1-\alpha)(m+1) \rceil / m}\!\left(\{R_j\}, \; \{w_j^{\mathrm{cal}}(x)\}\right)
$$

where the conformal weights are:

$$
w_j^{\mathrm{cal}}(x) = \frac{\tilde{w}_j(x)}{n_{\mathrm{eff}}^{\mathrm{cal}}(x)}, \quad
\tilde{w}_j(x) = \exp\left(-\frac{1}{2} \sum_{k=1}^{d} \frac{(x_k - x_{j,k}^{\mathrm{cal}})^2}{h_{\mathrm{cal},k}^2}\right)
$$

The bandwidth $h_{\mathrm{cal}}$ is **separate** from the point-predictor bandwidth $h$, because the optimal scales for prediction and calibration differ.

### 4.3 Weighted quantile algorithm

Given values $\{R_j\}$ and weights $\{w_j\}$:

1. Sort values ascending: $R_{(1)} \le R_{(2)} \le \dots \le R_{(m)}$, carrying weights.
2. Compute cumulative normalized weights: $c_k = \sum_{j \le k} w_{(j)} / \sum_j w_{(j)}$.
3. Return the smallest $R_{(k)}$ such that $c_k \ge q$.

Complexity: $O(m \log m)$ per query point.

### 4.4 Tuning $h_{\mathrm{cal}}$

$h_{\mathrm{cal}}$ is selected by grid search on a **validation set** (disjoint from both training and calibration). For each candidate $h_{\mathrm{cal}}$, we compute coverage and mean width on validation. The loss is:

$$
\mathcal{L}(h_{\mathrm{cal}}) = \begin{cases}
10^9 + \mathrm{width} & \text{if } \mathrm{coverage} < 1 - \alpha - 0.02 \\
\mathrm{width} & \text{otherwise}
\end{cases}
$$

That is, candidates that **undercover** by more than 2% are rejected; among valid candidates, the narrowest is chosen. The grid is $\log$-spaced in $[0.5, 10]$ (20 points).

**Why lower bound 0.5.** Below $h_{\mathrm{cal}} \approx 0.5$ in standardized coordinates, all kernel weights underflow to zero in float64, and weighted quantile degenerates to the nearest-neighbor residual. The lower bound prevents this.

---

## 5. Negative results

We document methods that were tested and did **not** work. This is included to save time for future researchers.

### 5.1 Soft-median via reweighting

An early version replaced the weighted mean with an iteratively reweighted soft-median:

$$
\tilde{w}_i \propto w_i \cdot \exp(-|y_i - \hat{y}| / \tau)
$$

Four ablations on 45 datasets (varying $\tau$ mode and IRLS iterations) showed:

| Method                                | Mean rank ↓   |
| ------------------------------------- | -------------- |
| τ = 10 (reweighting effectively off) | **2.31** |
| Auto τ (original)                    | 3.18           |
| Alternative auto τ                   | 3.0–3.5       |
| IRLS 5–10 iterations                 | 4.4–4.8       |

Reweighting **consistently degrades** performance. The best configuration is one where reweighting is essentially disabled. We therefore removed it entirely.

**Interpretation.** Soft-median converges to the median, which discards information about residual magnitude. On small data, every point carries information; robust reweighting is counterproductive.

### 5.2 Automatic temperature

The formula $\tau(x) = \delta_0(x) / \sqrt{n_{\mathrm{eff}}(x)}$ was tested against fixed $\tau$ and alternative auto-forms. All auto-forms were worse than $\tau = 10$ (effectively no reweighting). Removed.

### 5.3 Classification

SmallMLPClassifier was implemented with soft-vote + LOO-BCE. On 18 small binary classification datasets (hepatitis, parkinsons, sonar, ionosphere, breast cancer), SmallMLP ranked **5.6 / 8** on mean rank — behind Random Forest, MLPs, and logistic regression. A controlled study varying $d/n$ showed **no positive correlation** between dimensionality ratio and SmallMLP advantage (Pearson $r = -0.18$). We conclude that the soft-median mechanism, effective for regression, is not suited to classification. Classification support was removed.

---

## 6. Computational complexity

Let $n$ = training size, $m$ = calibration size, $d$ = feature dimension.

| Operation                     | Complexity                                                         |
| ----------------------------- | ------------------------------------------------------------------ |
| Kernel matrix$W$ (fit, LOO) | $O(n^2 d)$ per L-BFGS forward                                    |
| L-BFGS iteration              | ~10–20 kernel evaluations                                         |
| Total fit                     | $O(n^2 d \cdot I_{\text{LBFGS}})$                                |
| Point prediction              | $O(q \cdot n \cdot d)$ for $q$ queries                         |
| Weighted conformal            | $O(q \cdot m \log m)$                                            |
| $h_{\mathrm{cal}}$ tuning   | $O(G \cdot q_{\text{val}} \cdot m \log m)$ for $G$ grid points |

**Practical timings** (M2 MacBook Air, float64):

| n    | d  | Fit (s) | Predict (s, q=200) |
| ---- | -- | ------- | ------------------ |
| 100  | 5  | 0.036   | 0.0006             |
| 300  | 10 | 0.13    | 0.0015             |
| 500  | 20 | 0.62    | 0.0020             |
| 1000 | 20 | 1.57    | 0.0039             |

For the target regime $n < 500$, training takes under one second. Prediction is effectively instant.

---

## 7. Summary of contributions

1. **Learned bandwidth via LOO.** A vector bandwidth $h \in \mathbb{R}^d$ optimized by LOO-Huber loss with L-BFGS. Ablation shows this is the core mechanism (mean rank 1.44 vs 2.50 for fixed $h$).
2. **Weighted conformal intervals.** A novel application of weighted conformal prediction to small nonlinear regression, with per-feature learned calibration bandwidth. Reduces interval width by 19% relative to split conformal at equal coverage (31/32 wins).
3. **Honest negative results.** Soft-median reweighting, automatic temperature, and classification support were tested and rejected. This narrows the scope to what works.
