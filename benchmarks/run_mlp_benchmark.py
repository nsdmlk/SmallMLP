"""Benchmark: SmallMLP vs other MLPs on small regression datasets.

Hypothesis: SmallMLP beats standard MLPs without any hyperparameter tuning.
All baselines use default hyperparameters (no tuning) — same rule as SmallMLP.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_diabetes,
    make_regression,
    fetch_openml,
)
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from smallmlp import SmallMLPRegressor

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def build_datasets():
    datasets = []

    # 1. Synthetic: nonlinear, small n
    configs = [
        (80, 5, "sin"),
        (100, 10, "sin"),
        (150, 10, "sin"),
        (200, 20, "sin"),
        (300, 15, "poly"),
        (500, 20, "poly"),
    ]
    for i, (n, d, kind) in enumerate(configs):
        rng = np.random.default_rng(100 + i)
        X = rng.normal(size=(n, d))
        if kind == "sin":
            y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] + 0.1 * rng.normal(size=n)
        else:
            y = X[:, 0] ** 2 - X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=n)
        datasets.append((f"synth_{kind}_{n}x{d}", X, y))

    # 2. Real: diabetes (n=442, d=10)
    diab = load_diabetes()
    datasets.append(("diabetes", diab.data, diab.target))

    # 3. Real: subsampled diabetes at smaller n (true small-data regime)
    rng = np.random.default_rng(7)
    for n_sub in [50, 100, 200]:
        idx = rng.choice(len(diab.target), size=n_sub, replace=False)
        datasets.append((f"diabetes_{n_sub}", diab.data[idx], diab.target[idx]))

    # 4. OpenML small regression datasets (with fallback)
    openml_ids = {
        "concrete": 4353,       # n=1030, d=8
        "energy": 3080,         # n=768, d=8
        "yacht": 4556,          # n=308, d=6
        "airfoil": 44957,       # n=1503, d=5
        "wine_quality": 287,    # n=1599, d=11
    }
    for name, oml_id in openml_ids.items():
        try:
            ds = fetch_openml(data_id=oml_id, as_frame=False, parser="auto")
            X = ds.data.astype(float)
            y = ds.target.astype(float)
            # subsample to small-data regime
            for n_sub in [100, 300]:
                if len(y) >= n_sub:
                    idx = rng.choice(len(y), size=n_sub, replace=False)
                    datasets.append((f"{name}_{n_sub}", X[idx], y[idx]))
        except Exception as e:
            print(f"[skip] {name}: {e}")

    return datasets


# ---------------------------------------------------------------------------
# Models (all with default-ish hyperparameters, no tuning)
# ---------------------------------------------------------------------------

def build_models():
    return {
        # Baseline MLPs — sklearn defaults, no tuning
        "MLP_default": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(100,),
                max_iter=500,
                random_state=42,
            )),
        ]),
        "MLP_small": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(32,),
                max_iter=500,
                random_state=42,
            )),
        ]),
        "MLP_wide": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(256, 128),
                max_iter=500,
                random_state=42,
            )),
        ]),
        # Non-parametric baselines
        "KNN_k5": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=5)),
        ]),
        "KNN_k10": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=10)),
        ]),
        # SmallMLP: no preprocessing pipeline needed (it standardizes internally)
        "SmallMLP": SmallMLPRegressor(
            h_init=1.0,
            max_iter=100,
            tol=1e-8,
            verbose=False,
        ),
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(n_splits=5):
    datasets = build_datasets()
    models = build_models()
    model_names = list(models.keys())

    print(f"Datasets: {len(datasets)}")
    print(f"Models:   {len(model_names)}")
    print("=" * 70)

    rows = []
    for ds_name, X, y in datasets:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = {name: {"mae": [], "rmse": []} for name in model_names}

        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            for name, model in models.items():
                try:
                    # clone-like behavior: rebuild pipeline each fold
                    from sklearn.base import clone
                    m = clone(model)
                    m.fit(X_tr, y_tr)
                    y_pred = m.predict(X_te)
                    fold_results[name]["mae"].append(mean_absolute_error(y_te, y_pred))
                    fold_results[name]["rmse"].append(
                        np.sqrt(mean_squared_error(y_te, y_pred))
                    )
                except Exception as e:
                    fold_results[name]["mae"].append(np.nan)
                    fold_results[name]["rmse"].append(np.nan)

        for name in model_names:
            mae = np.nanmean(fold_results[name]["mae"])
            rmse = np.nanmean(fold_results[name]["rmse"])
            rows.append({
                "dataset": ds_name,
                "model": name,
                "mae": mae,
                "rmse": rmse,
                "n": len(y),
                "d": X.shape[1],
            })

        # quick progress
        best = min(
            model_names,
            key=lambda nm: np.nanmean(fold_results[nm]["mae"])
        )
        print(f"{ds_name:28s}  best={best:12s}  "
              f"SmallMLP_MAE={np.nanmean(fold_results['SmallMLP']['mae']):.4f}")

    df = pd.DataFrame(rows)
    return df


def summarize(df):
    print("\n" + "=" * 70)
    print("SUMMARY: mean MAE across datasets (lower is better)")
    print("=" * 70)

    pivot = df.pivot(index="dataset", columns="model", values="mae")
    # rank models per dataset, then average ranks
    ranks = pivot.rank(axis=1, method="average")
    mean_ranks = ranks.mean(axis=0).sort_values()

    print("\nMean rank (1 = best):")
    for name, r in mean_ranks.items():
        print(f"  {name:15s}: {r:.3f}")

    print("\nMean MAE:")
    mean_mae = pivot.mean(axis=0).sort_values()
    for name, m in mean_mae.items():
        print(f"  {name:15s}: {m:.4f}")

    print("\nWin count (best MAE per dataset):")
    wins = (pivot == pivot.min(axis=1).values[:, None]).sum(axis=0).sort_values(
        ascending=False
    )
    for name, w in wins.items():
        print(f"  {name:15s}: {int(w)}")

    return pivot, mean_ranks, mean_mae, wins


if __name__ == "__main__":
    df = run_benchmark(n_splits=5)
    df.to_csv("benchmarks/results_mlp.csv", index=False)
    pivot, ranks, mae, wins = summarize(df)
    pivot.to_csv("benchmarks/pivot_mae.csv")
    print("\nSaved: benchmarks/results_mlp.csv, benchmarks/pivot_mae.csv")