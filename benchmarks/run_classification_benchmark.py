"""
ReLU bottleneck sweep.

For each dataset, train SmallMLPClassifier with ReLU and measure:
  T1 — dead units per layer (fraction of units that output 0 for all inputs)
  T2 — bias negativity (fraction of negative biases per layer)
  T3 — dead-zone inputs (fraction of pre-activations < -2)
  T4 — pre-activation distribution (mean, std, skewness)
  T5 — gradient norm per layer at end of training

Goal: find which of these correlates with low accuracy.
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from smallmlp import SmallMLPClassifier
from run_full_benchmark import build_classification_datasets

warnings.filterwarnings("ignore")


def measure_relu(clf, X_np, y_np):
    """Return diagnostic metrics for a trained ReLU model."""
    X = torch.tensor(StandardScaler().fit_transform(X_np), dtype=torch.float32)
    model = clf._model
    model.eval()

    metrics = {"n_layers": len(model.layers)}
    h = X
    with torch.no_grad():
        for i, (fc, act, drop) in enumerate(zip(model.layers, model.acts, model.drops)):
            pre = fc(h)
            post = act(pre)

            # T1: dead units (output 0 for all inputs)
            dead_frac = (post.abs().max(dim=0).values < 1e-6).float().mean().item()

            # T2: bias negativity
            bias = fc.bias.detach()
            bias_neg_frac = (bias < 0).float().mean().item()
            bias_mean = bias.mean().item()

            # T3: dead-zone inputs
            deadzone_frac = (pre < -2).float().mean().item()

            # T4: pre-activation distribution
            pre_flat = pre.flatten()
            pre_mean = pre_flat.mean().item()
            pre_std = pre_flat.std().item()
            pre_skew = ((pre_flat - pre_mean) ** 3).mean().item() / (pre_std ** 3 + 1e-9)

            # T5: gradient norm (zero-grad fraction on ReLU)
            zero_grad_frac = ((pre < 0).float()).mean().item()

            metrics[f"L{i+1}_dead"] = dead_frac
            metrics[f"L{i+1}_bias_neg"] = bias_neg_frac
            metrics[f"L{i+1}_bias_mean"] = bias_mean
            metrics[f"L{i+1}_deadzone"] = deadzone_frac
            metrics[f"L{i+1}_pre_mean"] = pre_mean
            metrics[f"L{i+1}_pre_std"] = pre_std
            metrics[f"L{i+1}_pre_skew"] = pre_skew
            metrics[f"L{i+1}_zero_grad_frac"] = zero_grad_frac

            h = post

    return metrics


def run_diagnostic(datasets, n_splits=5, seed=42):
    rows = []
    for name, X, y in datasets:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        accs = []
        fold_metrics = []
        for tr, te in skf.split(X, y):
            clf = SmallMLPClassifier(
                activation="relu", width_mode="formula",
                class_weight=None, alpha=4.0, beta=0.7,
                max_epochs=500, patience=30, random_state=seed,
            )
            try:
                clf.fit(X[tr], y[tr])
                acc = accuracy_score(y[te], clf.predict(X[te]))
                metrics = measure_relu(clf, X[tr], y[tr])
            except Exception as e:
                warnings.warn(f"{name}: {e}")
                acc = np.nan
                metrics = {}
            accs.append(acc)
            if metrics:
                fold_metrics.append(metrics)

        row = {"dataset": name, "acc": float(np.nanmean(accs))}
        if fold_metrics:
            for k in fold_metrics[0]:
                vals = [m[k] for m in fold_metrics if k in m]
                row[k] = float(np.nanmean(vals))
        rows.append(row)
    return rows


def main():
    datasets = build_classification_datasets()
    datasets = [(n, X, y) for n, X, y in datasets if np.bincount(y).min() >= 5]
    print(f"Using {len(datasets)} datasets")

    rows = run_diagnostic(datasets, n_splits=5, seed=42)

    # print table of key metrics
    print()
    keys = ["acc",
            "L1_dead", "L1_bias_neg", "L1_deadzone", "L1_pre_std", "L1_pre_skew", "L1_zero_grad_frac",
            "L2_dead", "L2_bias_neg", "L2_deadzone", "L2_pre_std", "L2_pre_skew", "L2_zero_grad_frac"]

    print(f"{'dataset':<22} " + " ".join(f"{k:>10}" for k in keys))
    for row in rows:
        line = f"{row['dataset']:<22} "
        for k in keys:
            v = row.get(k, np.nan)
            line += f"{v:>10.3f} "
        print(line)

    # correlations with accuracy
    print()
    print("Correlation of each metric with accuracy (across datasets):")
    accs = np.array([r["acc"] for r in rows])
    for k in keys:
        if k == "acc":
            continue
        vals = np.array([r.get(k, np.nan) for r in rows])
        mask = np.isfinite(vals) & np.isfinite(accs)
        if mask.sum() < 3:
            continue
        corr = np.corrcoef(vals[mask], accs[mask])[0, 1]
        print(f"  {k:<20} corr={corr:+.3f}")

    # identify worst datasets
    print()
    print("Worst 5 datasets by accuracy:")
    sorted_rows = sorted(rows, key=lambda r: r["acc"])
    for r in sorted_rows[:5]:
        print(f"  {r['dataset']:<22} acc={r['acc']:.4f}  "
              f"L1_dead={r.get('L1_dead', np.nan):.3f}  "
              f"L1_deadzone={r.get('L1_deadzone', np.nan):.3f}  "
              f"L1_bias_neg={r.get('L1_bias_neg', np.nan):.3f}  "
              f"L2_deadzone={r.get('L2_deadzone', np.nan):.3f}")

    # save
    import pandas as pd
    pd.DataFrame(rows).to_csv("benchmarks/relu_diagnostic.csv", index=False)
    print("\nSaved to benchmarks/relu_diagnostic.csv")


if __name__ == "__main__":
    main()