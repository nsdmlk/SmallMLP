"""Ablation: does iterative IRLS help soft-median?"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from smallmlp import SmallMLPRegressor
from benchmarks.run_mlp_benchmark import build_datasets

warnings.filterwarnings("ignore")


def run_ablation(n_splits=5):
    datasets = build_datasets()

    variants = {
        "irls1_auto":  {"n_irls": 1,  "tau_mode": "auto"},
        "irls3_auto":  {"n_irls": 3,  "tau_mode": "auto"},
        "irls5_auto":  {"n_irls": 5,  "tau_mode": "auto"},
        "irls10_auto": {"n_irls": 10, "tau_mode": "auto"},
        "irls1_tau10": {"n_irls": 1,  "tau_mode": 10.0},
        "irls5_tau10": {"n_irls": 5,  "tau_mode": 10.0},
    }

    print(f"Datasets: {len(datasets)}")
    print(f"Variants: {list(variants.keys())}")
    print("=" * 70)

    rows = []
    for ds_name, X, y in datasets:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = {name: [] for name in variants}

        for tr, te in kf.split(X):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]

            for name, cfg in variants.items():
                try:
                    m = SmallMLPRegressor(
                        h_min=0.01, h_max=10.0,
                        max_iter=30, inner_iter=10, tol=1e-8,
                        tau_mode=cfg["tau_mode"],
                        n_irls=cfg["n_irls"],
                    )
                    m.fit(X_tr, y_tr)
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

        best = min(variants, key=lambda nm: np.nanmean(fold_results[nm]))
        print(f"{ds_name:28s}  best={best:14s}  "
              f"mae={np.nanmean(fold_results[best]):7.4f}")

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="dataset", columns="model", values="mae")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nMean MAE:")
    for name, m in pivot.mean(axis=0).sort_values().items():
        print(f"  {name:14s}: {m:.4f}")

    print("\nMean rank (1 = best):")
    for name, r in pivot.rank(axis=1, method="average").mean(axis=0).sort_values().items():
        print(f"  {name:14s}: {r:.3f}")

    print("\nWin count:")
    wins = (pivot == pivot.min(axis=1).values[:, None]).sum(axis=0).sort_values(
        ascending=False
    )
    for name, w in wins.items():
        print(f"  {name:14s}: {int(w)}")

    return pivot


if __name__ == "__main__":
    pivot = run_ablation(n_splits=5)
    pivot.to_csv("benchmarks/ablation_irls.csv")
    print("\nSaved: benchmarks/ablation_irls.csv")