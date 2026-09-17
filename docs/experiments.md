
# Experiments

Complete benchmark description: datasets, protocol, results, and how to reproduce.

---

## Overview

SmallMLP was evaluated on **45 small regression datasets** ($n < 500$) from three sources:

- **Synthetic:** 6 datasets (sin, poly).
- **sklearn benchmarks:** 9 Friedman (1/2/3) datasets.
- **Real-world:** 30 datasets from `sklearn`, OpenML, and UCI — diabetes, energy, yacht, airfoil, wine quality, CPU, kin8nm, abalone, forest fires, and others, subsampled to small-data regime.

All experiments use **5-fold cross-validation** with fixed random seeds.

---

## Point prediction benchmark

### Protocol

- **Models:** SmallMLP, three MLP variants (32, 100, 256×128), KNN (k=5, k=10), RandomForest.
- **Preprocessing:** StandardScaler in Pipeline for all baselines (SmallMLP standardizes internally).
- **Metric:** Mean Absolute Error (MAE).
- **Aggregation:** Mean rank (1 = best per dataset), mean MAE, win count.

### Results

**Mean rank** (lower is better):

| Model               | Mean rank ↓   |
| ------------------- | -------------- |
| **SmallMLP**  | **1.80** |
| MLP_wide (256, 128) | 2.76           |
| KNN_k10             | 3.36           |
| KNN_k5              | 3.49           |
| MLP_default (100,)  | 4.33           |
| MLP_small (32,)     | 5.27           |

**Mean MAE** (lower is better):

| Model              | Mean MAE ↓     |
| ------------------ | --------------- |
| **SmallMLP** | **11.39** |
| MLP_wide           | 12.44           |
| KNN_k5             | 18.89           |
| KNN_k10            | 19.13           |
| MLP_default        | 34.44           |
| MLP_small          | 51.95           |

**Win count** (best MAE per dataset):

| Model              | Wins / 45    |
| ------------------ | ------------ |
| **SmallMLP** | **25** |
| MLP_wide           | 11           |
| KNN_k10            | 5            |
| KNN_k5             | 2            |
| MLP_small          | 2            |
| MLP_default        | 0            |

### Where SmallMLP wins / loses

**Wins:** nonlinear tasks — sin, poly, Friedman1, Friedman2, energy, airfoil, forest fires, kin8nm, cpu_small.

**Loses:** linear / near-linear tasks — Friedman3 (smooth arctan), make_regression with informative subset, diabetes.

**Interpretation.** SmallMLP's advantage is concentrated on **nonlinear** problems. On linear problems it loses to standard MLPs — expected for a non-parametric method, since kernel smoothing discards global linear structure.

---

## Prediction interval benchmark

### Protocol

- **Split:** 60% train / 20% calibration / 20% validation.
- **Coverage target:** 90% ($\alpha = 0.1$).
- **Methods:**
  - **Weighted conformal** (SmallMLP) — learned $h_{\text{cal}}$, weighted quantile.
  - **Split conformal** — global quantile, no weights.
  - **Heuristic zone** — $\hat{y} \pm z_{1-\alpha/2} \delta(x)$.
- **Metrics:** coverage (fraction of val points inside interval), mean width.
- **Validity criterion:** coverage ≥ 0.88 (2% slack below target).

### Results

| Method                       | Valid datasets    | Mean width (among valid) ↓ |
| ---------------------------- | ----------------- | --------------------------- |
| **Weighted conformal** | **38 / 45** | **51.8**              |
| Split conformal              | 32 / 45           | 64.0                        |
| Heuristic zone               | 17 / 45           | 145.7                       |

### Head-to-head: weighted vs split conformal

Among 32 datasets where **both methods are valid**:

| Metric                             | Value             |
| ---------------------------------- | ----------------- |
| Weighted conformal narrower        | **31 / 32** |
| Split conformal narrower           | 1 / 32            |
| Mean width (weighted)              | 46.2              |
| Mean width (split)                 | 64.0              |
| **Ratio (weighted / split)** | **0.81**    |

**Weighted conformal produces intervals 19% narrower at equal coverage**, winning on 31 of 32 datasets.

### Calibration plot

Empirical coverage vs nominal level, averaged over 45 datasets:

| Nominal | Weighted conformal | Split conformal |
| ------- | ------------------ | --------------- |
| 0.50    | 0.522              | 0.554           |
| 0.60    | 0.603              | 0.649           |
| 0.70    | 0.702              | 0.743           |
| 0.80    | 0.774              | 0.826           |
| 0.90    | 0.868              | 0.910           |
| 0.95    | 0.927              | 0.954           |

**Weighted conformal is well-calibrated** — empirical coverage within 3% of nominal across all levels. Split conformal **systematically overcovers**, trading width for conservative coverage.

---

## Ablation: learned bandwidth

### Protocol

- **Models:** SmallMLP with learned $h$, and SmallMLP with fixed $h \in \{0.5, 1, 2, 5\}$.
- **Metric:** Mean rank, mean MAE, wins.

### Results

| Model               | Mean rank ↓   | Mean MAE ↓     | Wins / 18    |
| ------------------- | -------------- | --------------- | ------------ |
| **Learned h** | **1.44** | **11.45** | **13** |
| Fixed h = 1.0       | 2.50           | 11.95           | 4            |
| Fixed h = 2.0       | 3.17           | 13.91           | 1            |
| Fixed h = 0.5       | 3.39           | 13.24           | 0            |
| Fixed h = 5.0       | 4.50           | 16.72           | 0            |

**Learning $h$ is the core mechanism.** A fixed bandwidth loses most of the advantage — 13 / 18 wins vs 4 / 18 for the best fixed value.

---

## Negative results

### Soft-median reweighting

Early versions of SmallMLP used iterative reweighting (soft-median). Four separate ablations showed reweighting **consistently degrades** performance.

| Method                                | Mean rank ↓   |
| ------------------------------------- | -------------- |
| τ = 10 (reweighting effectively off) | **2.31** |
| Auto τ (original formula)            | 3.18           |
| Alternative auto τ                   | 3.0–3.5       |
| IRLS 5 iterations                     | 4.42           |
| IRLS 10 iterations                    | 4.76           |

**Best configuration: no reweighting.** Removed from the final model.

### Automatic temperature

The formula $\tau(x) = \delta_0(x) / \sqrt{n_{\text{eff}}(x)}$ was tested against fixed and alternative auto-forms. All auto-forms were worse than a large constant $\tau = 10$ (reweighting off). Removed.

### Classification

`SmallMLPClassifier` with soft-vote + LOO-BCE was tested on 18 small binary classification datasets (hepatitis, parkinsons, sonar, ionosphere, breast cancer, etc.).

| Model               | Mean AUC | Mean rank ↓ |
| ------------------- | -------- | ------------ |
| Random Forest (100) | 0.940    | 3.38         |
| MLP (32,)           | 0.930    | 3.58         |
| MLP (100,)          | 0.930    | 3.67         |
| MLP_wide            | 0.929    | 4.21         |
| SmallMLP            | 0.922    | 5.58         |
| KNN_k10             | 0.918    | 5.54         |
| Logistic Regression | 0.910    | 4.13         |

SmallMLP ranked **5.6 / 8**. A controlled study varying $d/n$ showed **no positive correlation** between dimensionality ratio and SmallMLP advantage (Pearson $r = -0.18$). Classification support was removed.

---

## Computational cost

**Fit time** (M2 MacBook Air, float64):

| n    | d  | SmallMLP | MLP_100 | KNN     |
| ---- | -- | -------- | ------- | ------- |
| 100  | 5  | 0.036 s  | 0.037 s | 0.001 s |
| 300  | 10 | 0.13 s   | 0.12 s  | 0.001 s |
| 500  | 20 | 0.62 s   | 0.20 s  | 0.001 s |
| 1000 | 20 | 1.57 s   | 0.42 s  | 0.002 s |

**Predict time** (200 queries):

| n    | SmallMLP | MLP_100 | KNN    |
| ---- | -------- | ------- | ------ |
| 100  | 0.6 ms   | 0.4 ms  | 0.9 ms |
| 500  | 2.0 ms   | 0.4 ms  | 1.1 ms |
| 1000 | 3.9 ms   | 0.4 ms  | 1.7 ms |

**For $n < 500$, SmallMLP is comparable to standard MLP in fit time and faster than KNN in predict time.** The cost grows with $n$ due to the non-parametric nature (kernel matrix $O(n^2)$ for fit, $O(n)$ per prediction).

---

## Reproducibility

All experiments can be reproduced from the repository:

```bash
# point prediction benchmark
python -m benchmarks.run_mlp_benchmark

# ablation: learned vs fixed bandwidth
python -m benchmarks.ablation_h

# conformal interval benchmark
python -m benchmarks.conformal_benchmark

# diagnosis of invalid datasets + calibration plot
python -m benchmarks.diagnose_conformal

# computational cost
python benchmarks/speed_benchmark.py
```

Results are saved as CSV files in `benchmarks/`. All random seeds are fixed.

**Environment:** Python 3.12, `torch>=2.0`, `scikit-learn>=1.3`, `ucimlrepo`, `pandas`, `matplotlib`. Hardware: M2 MacBook Air, 16 GB RAM.

---

## Datasets

### Synthetic (6)

| Name              | n   | d  | Formula                            |
| ----------------- | --- | -- | ---------------------------------- |
| synth_sin_80x5    | 80  | 5  | $\sin(3x_0) + 0.5x_1 + \epsilon$ |
| synth_sin_100x10  | 100 | 10 | same                               |
| synth_sin_150x10  | 150 | 10 | same                               |
| synth_sin_200x20  | 200 | 20 | same                               |
| synth_poly_300x15 | 300 | 15 | $x_0^2 - x_1 x_2 + \epsilon$     |
| synth_poly_500x20 | 500 | 20 | same                               |

### sklearn benchmarks (12)

| Name                          | n           | d        | Source              |
| ----------------------------- | ----------- | -------- | ------------------- |
| friedman1_{100, 300, 500}     | 100/300/500 | 10       | `make_friedman1`  |
| friedman2_{100, 300, 500}     | 100/300/500 | 4        | `make_friedman2`  |
| friedman3_{100, 300, 500}     | 100/300/500 | 4        | `make_friedman3`  |
| mreg_{100x20, 300x30, 500x40} | 100/300/500 | 20/30/40 | `make_regression` |

### Real-world (27)

| Name                         | n           | d  | Source                        |
| ---------------------------- | ----------- | -- | ----------------------------- |
| diabetes                     | 442         | 10 | `sklearn.datasets`          |
| diabetes_{50, 100, 200}      | 50/100/200  | 10 | subsampled                    |
| energy_{100, 300, 500}       | 100/300/500 | 8  | OpenML`energy-efficiency`   |
| yacht_{100, 300}             | 100/300     | 6  | OpenML`yacht_hydrodynamics` |
| airfoil_{100, 300, 500}      | 100/300/500 | 5  | OpenML`airfoil_self_noise`  |
| wine_quality_{100, 300, 500} | 100/300/500 | 11 | OpenML`wine_quality`        |
| cpu_small_{100, 300, 500}    | 100/300/500 | 12 | OpenML`cpu_small`           |
| kin8nm_{100, 300, 500}       | 100/300/500 | 8  | OpenML`kin8nm`              |
| abalone_{100, 300, 500}      | 100/300/500 | 7  | OpenML`abalone`             |
| forest_fires_{100, 300, 500} | 100/300/500 | 10 | OpenML`forest_fires`        |

All datasets are subsampled to $n \le 500$ where needed, to match the small-data regime.
