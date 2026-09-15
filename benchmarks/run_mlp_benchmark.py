"""Benchmark: SmallMLP vs other MLPs on small regression datasets.

Hypothesis: SmallMLP beats standard MLPs without any hyperparameter tuning.
All baselines use default hyperparameters (no tuning) — same rule as SmallMLP.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_openml
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from smallmlp import SmallMLPRegressor

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _load_openml_regression(name, version=1):
    """Load a regression dataset from OpenML as (X, y) float arrays.

    Handles multi-target datasets (e.g. energy-efficiency) by taking the
    first numeric target column.
    """
    ds = fetch_openml(name=name, version=version, as_frame=True, parser="auto")
    X_df = ds.data.select_dtypes(include=[np.number])
    X = X_df.to_numpy(dtype=float)

    target = ds.target
    if hasattr(target, "columns"):          # multi-target frame
        y = target[target.columns[0]].to_numpy(dtype=float)
    else:
        y = np.asarray(target, dtype=float)

    # final safety: drop non-finite
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask], y[mask]


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

    # 3. Subsampled diabetes (true small-data regime)
    rng = np.random.default_rng(7)
    for n_sub in [50, 100, 200]:
        idx = rng.choice(len(diab.target), size=n_sub, replace=False)
        datasets.append((f"diabetes_{n_sub}", diab.data[idx], diab.target[idx]))

    # 4. OpenML small regression datasets
    openml_names = {
        "energy": "energy-efficiency",
        "yacht": "yacht_hydrodynamics",
        "airfoil": "airfoil_self_noise",
        "wine_quality": "wine_quality",
    }
    for name, oml_name in openml_names.items():
        try:
            X, y = _load_openml_regression(oml_name)
            print(f"[ok]   {name}: n={len(y)}, d={X.shape[1]}")
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
        "KNN_k5": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=5)),
        ]),
        "KNN_k10": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=10)),
        ]),
        "SmallMLP": SmallMLPRegressor(
            h_min=0.01,
            h_max=10.0,
            max_iter=30,
            inner_iter=10,
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

    print(f"\nDatasets: {len(datasets)}")
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
                    m = clone(model)
                    m.fit(X_tr, y_tr)
                    y_pred = m.predict(X_te)
                    fold_results[name]["mae"].append(
                        mean_absolute_error(y_te, y_pred)
                    )
                    fold_results[name]["rmse"].append(
                        np.sqrt(mean_squared_error(y_te, y_pred))
                    )
                except Exception as e:
                    fold_results[name]["mae"].append(np.nan)
                    fold_results[name]["rmse"].append(np.nan)

        for name in model_names:
            rows.append({
                "dataset": ds_name,
                "model": name,
                "mae": np.nanmean(fold_results[name]["mae"]),
                "rmse": np.nanmean(fold_results[name]["rmse"]),
                "n": len(y),
                "d": X.shape[1],
            })

        best = min(
            model_names,
            key=lambda nm: np.nanmean(fold_results[nm]["mae"])
        )
        print(f"{ds_name:28s}  best={best:12s}  "
              f"SmallMLP_MAE={np.nanmean(fold_results['SmallMLP']['mae']):.4f}")

    return pd.DataFrame(rows)


def summarize(df):
    print("\n" + "=" * 70)
    print("SUMMARY: mean MAE across datasets (lower is better)")
    print("=" * 70)

    pivot = df.pivot(index="dataset", columns="model", values="mae")
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