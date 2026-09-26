"""Regression benchmark: Adaptive MLP vs baselines.

45 small regression datasets (n < 500), 5-fold CV, MAE.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_openml
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from smallmlp.nn import AdaptiveMLPRegressor

warnings.filterwarnings("ignore")


def _load_openml_regression(name, version=1):
    ds = fetch_openml(name=name, version=version, as_frame=True, parser="auto")
    X_df = ds.data.select_dtypes(include=[np.number])
    X = X_df.to_numpy(dtype=float)
    target = ds.target
    if hasattr(target, "columns"):
        y = target[target.columns[0]].to_numpy(dtype=float)
    else:
        y = np.asarray(target, dtype=float)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask], y[mask]


def build_datasets():
    datasets = []

    # synthetic
    configs = [
        (80, 5, "sin"), (100, 10, "sin"), (150, 10, "sin"),
        (200, 20, "sin"), (300, 15, "poly"), (500, 20, "poly"),
    ]
    for i, (n, d, kind) in enumerate(configs):
        rng = np.random.default_rng(100 + i)
        X = rng.normal(size=(n, d))
        if kind == "sin":
            y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] + 0.1 * rng.normal(size=n)
        else:
            y = X[:, 0] ** 2 - X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=n)
        datasets.append((f"synth_{kind}_{n}x{d}", X, y))

    # friedman
    from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3
    for n in [100, 300, 500]:
        for name, fn in [("friedman1", make_friedman1),
                         ("friedman2", make_friedman2),
                         ("friedman3", make_friedman3)]:
            try:
                X, y = fn(n_samples=n, noise=0.1, random_state=42)
                datasets.append((f"{name}_{n}", X, y))
            except Exception as e:
                print(f"[skip] {name}_{n}: {e}")

    # make_regression
    from sklearn.datasets import make_regression
    for n, d, n_inf in [(100, 20, 5), (300, 30, 8), (500, 40, 10)]:
        X, y = make_regression(
            n_samples=n, n_features=d, n_informative=n_inf,
            noise=0.3, random_state=42,
        )
        datasets.append((f"mreg_{n}x{d}_inf{n_inf}", X, y))

    # diabetes
    diab = load_diabetes()
    datasets.append(("diabetes", diab.data, diab.target))
    rng = np.random.default_rng(7)
    for n_sub in [50, 100, 200]:
        idx = rng.choice(len(diab.target), size=n_sub, replace=False)
        datasets.append((f"diabetes_{n_sub}", diab.data[idx], diab.target[idx]))

    # OpenML
    openml_names = {
        "energy": "energy-efficiency",
        "yacht": "yacht_hydrodynamics",
        "airfoil": "airfoil_self_noise",
        "wine_quality": "wine_quality",
        "cpu_small": "cpu_small",
        "kin8nm": "kin8nm",
        "abalone": "abalone",
        "forest_fires": "forest_fires",
    }
    for name, oml_name in openml_names.items():
        try:
            X, y = _load_openml_regression(oml_name)
            print(f"[ok] {name}: n={len(y)}, d={X.shape[1]}")
            for n_sub in [100, 300, 500]:
                if len(y) >= n_sub:
                    idx = rng.choice(len(y), size=n_sub, replace=False)
                    datasets.append((f"{name}_{n_sub}", X[idx], y[idx]))
        except Exception as e:
            print(f"[skip] {name}: {e}")

    return datasets


def build_models():
    return {
        "MLP_32": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)),
        ]),
        "MLP_100": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)),
        ]),
        "MLP_wide": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=1000, random_state=42)),
        ]),
        "KNN_k5": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=5)),
        ]),
        "KNN_k10": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=10)),
        ]),
        "RF_100": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "AdaptiveMLP": AdaptiveMLPRegressor(
            max_epochs=500, patience=30, verbose=False,
        ),
    }


def run_benchmark(n_splits=5):
    datasets = build_datasets()
    models = build_models()

    print(f"\nDatasets: {len(datasets)}")
    print(f"Models:   {len(models)}")
    print("=" * 100)

    rows = []
    for ds_name, X, y in datasets:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        maes = {name: [] for name in models}
        for tr, te in kf.split(X):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            for name, model in models.items():
                try:
                    m = clone(model)
                    m.fit(X_tr, y_tr)
                    y_pred = m.predict(X_te)
                    maes[name].append(mean_absolute_error(y_te, y_pred))
                except Exception:
                    maes[name].append(np.nan)
        for name in models:
            rows.append({
                "dataset": ds_name, "model": name,
                "mae": np.nanmean(maes[name]),
                "n": len(y), "d": X.shape[1],
            })
        best = min(models, key=lambda nm: np.nanmean(maes[nm]))
        print(f"{ds_name:28s}  best={best:14s}  "
              f"AdaptiveMLP_MAE={np.nanmean(maes['AdaptiveMLP']):.4f}")

    return pd.DataFrame(rows)


def summarize(df):
    print("\n" + "=" * 100)
    print("SUMMARY: regression")
    print("=" * 100)
    pivot = df.pivot(index="dataset", columns="model", values="mae")

    print("\nMean MAE:")
    for name, m in pivot.mean(axis=0).sort_values().items():
        print(f"  {name:14s}: {m:.4f}")

    print("\nMean rank (1 = best):")
    ranks = pivot.rank(axis=1, method="average")
    for name, r in ranks.mean(axis=0).sort_values().items():
        print(f"  {name:14s}: {r:.3f}")

    print("\nWin count:")
    wins = (pivot == pivot.min(axis=1).values[:, None]).sum(axis=0)
    for name, w in wins.sort_values(ascending=False).items():
        print(f"  {name:14s}: {int(w)}")
    return pivot


if __name__ == "__main__":
    df = run_benchmark(n_splits=5)
    df.to_csv("benchmarks/results_regression.csv", index=False)
    summarize(df)
    print("\nSaved: benchmarks/results_regression.csv")