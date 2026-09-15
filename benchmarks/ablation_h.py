"""Ablation: does training h actually help?

Compares SmallMLP with learned h against variants with fixed h values.
If learned h is not better, the model's core claim is weakened.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

from smallmlp import SmallMLPRegressor
from benchmarks.run_mlp_benchmark import build_datasets  # reuse loaders

warnings.filterwarnings("ignore")


def make_frozen_h_model(h_value):
    model = SmallMLPRegressor(
        h_min=0.01, h_max=10.0,
        max_iter=0,
        inner_iter=0,
        verbose=False,
    )
    model._frozen_h = h_value
    return model


def fit_frozen(model, X, y):
    from sklearn.utils.validation import check_X_y
    import torch

    X, y = check_X_y(X, y, dtype=np.float64)
    model.n_features_in_ = X.shape[1]
    model._x_mean = X.mean(axis=0)
    model._x_std = X.std(axis=0)
    model._x_std[model._x_std == 0] = 1.0
    model._y_mean = float(y.mean())
    y_std = float(y.std())
    model._y_std = y_std if y_std > 0 else 1.0

    Xs = (X - model._x_mean) / model._x_std
    ys = (y - model._y_mean) / model._y_std
    model._X_train = torch.tensor(Xs, dtype=torch.float64)
    model._y_train = torch.tensor(ys, dtype=torch.float64)

    # inverse of h = h_min + (h_max - h_min) * sigmoid(psi)
    p = (model._frozen_h - model.h_min) / (model.h_max - model.h_min)
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    psi_val = np.log(p / (1 - p))
    model._psi = torch.tensor(
        np.full(model.n_features_in_, psi_val), dtype=torch.float64
    )
    model._loss_ = None
    return model


def run_ablation(n_splits=5):
    datasets = build_datasets()
    h_values = [0.5, 1.0, 2.0, 5.0]

    variants = {"SmallMLP_trained": None}
    for hv in h_values:
        variants[f"SmallMLP_h{hv}"] = hv

    print(f"Datasets: {len(datasets)}")
    print(f"Variants: {list(variants.keys())}")
    print("=" * 70)

    rows = []
    for ds_name, X, y in datasets:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = {name: [] for name in variants}

        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # trained
            try:
                m = SmallMLPRegressor(
                    h_min=0.01, h_max=10.0,
                    max_iter=30, inner_iter=10, tol=1e-8,
                )
                m.fit(X_tr, y_tr)
                y_pred = m.predict(X_te)
                fold_results["SmallMLP_trained"].append(
                    mean_absolute_error(y_te, y_pred)
                )
            except Exception:
                fold_results["SmallMLP_trained"].append(np.nan)

            # fixed h
            for hv in h_values:
                name = f"SmallMLP_h{hv}"
                try:
                    m = make_frozen_h_model(hv)
                    fit_frozen(m, X_tr, y_tr)
                    y_pred = m.predict(X_te)
                    fold_results[name].append(mean_absolute_error(y_te, y_pred))
                except Exception:
                    fold_results[name].append(np.nan)

        for name in variants:
            rows.append({
                "dataset": ds_name,
                "model": name,
                "mae": np.nanmean(fold_results[name]),
            })

        best_fixed = min(
            (f"SmallMLP_h{hv}" for hv in h_values),
            key=lambda nm: np.nanmean(fold_results[nm]),
        )
        print(f"{ds_name:28s}  "
              f"trained={np.nanmean(fold_results['SmallMLP_trained']):7.4f}  "
              f"best_fixed={best_fixed} ({np.nanmean(fold_results[best_fixed]):7.4f})")

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="dataset", columns="model", values="mae")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nMean MAE:")
    for name, m in pivot.mean(axis=0).sort_values().items():
        print(f"  {name:22s}: {m:.4f}")

    print("\nMean rank (1 = best):")
    for name, r in pivot.rank(axis=1, method="average").mean(axis=0).sort_values().items():
        print(f"  {name:22s}: {r:.3f}")

    print("\nWin count:")
    wins = (pivot == pivot.min(axis=1).values[:, None]).sum(axis=0).sort_values(
        ascending=False
    )
    for name, w in wins.items():
        print(f"  {name:22s}: {int(w)}")

    return pivot


if __name__ == "__main__":
    pivot = run_ablation(n_splits=5)
    pivot.to_csv("benchmarks/ablation_h.csv")
    print("\nSaved: benchmarks/ablation_h.csv")