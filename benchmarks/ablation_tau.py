"""Ablation: does automatic tau help?

Compares tau = delta0 / sqrt(n_eff) (auto) vs fixed constants.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

from smallmlp import SmallMLPRegressor
from benchmarks.run_mlp_benchmark import build_datasets

warnings.filterwarnings("ignore")


def run_ablation(n_splits=5):
    datasets = build_datasets()

    variants = {
        "tau_auto": "auto",
        "tau_const1": "const",
        "tau_0.1": 0.1,
        "tau_10": 10.0,
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

            for name, tm in variants.items():
                try:
                    m = SmallMLPRegressor(
                        h_min=0.01, h_max=10.0,
                        max_iter=30, inner_iter=10, tol=1e-8,
                        tau_mode=tm,
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

        auto_mae = np.nanmean(fold_results["tau_auto"])
        best_fixed_name = min(
            [n for n in variants if n != "tau_auto"],
            key=lambda nm: np.nanmean(fold_results[nm]),
        )
        best_fixed = np.nanmean(fold_results[best_fixed_name])
        marker = " +" if auto_mae < best_fixed else ""
        print(f"{ds_name:28s}  auto={auto_mae:7.4f}  "
              f"best_fixed={best_fixed_name}({best_fixed:7.4f}){marker}")

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="dataset", columns="model", values="mae")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nMean MAE:")
    for name, m in pivot.mean(axis=0).sort_values().items():
        print(f"  {name:12s}: {m:.4f}")

    print("\nMean rank (1 = best):")
    for name, r in pivot.rank(axis=1, method="average").mean(axis=0).sort_values().items():
        print(f"  {name:12s}: {r:.3f}")

    print("\nWin count:")
    wins = (pivot == pivot.min(axis=1).values[:, None]).sum(axis=0).sort_values(
        ascending=False
    )
    for name, w in wins.items():
        print(f"  {name:12s}: {int(w)}")

    return pivot


if __name__ == "__main__":
    pivot = run_ablation(n_splits=5)
    pivot.to_csv("benchmarks/ablation_tau.csv")
    print("\nSaved: benchmarks/ablation_tau.csv")