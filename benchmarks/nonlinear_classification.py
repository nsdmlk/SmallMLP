"""Nonlinear classification test: two moons, circles, XOR, spirals.

Quick check whether SmallMLP's locality helps on genuinely nonlinear
decision boundaries, where UCI datasets failed.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from smallmlp import SmallMLPClassifier

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Synthetic nonlinear datasets
# ---------------------------------------------------------------------------

def make_xor(n=400, noise=0.15, random_state=0):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    # flip labels with prob = noise
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return X, y


def make_spirals(n=400, noise=0.1, random_state=0):
    """Two interleaved spirals."""
    rng = np.random.default_rng(random_state)
    n_per = n // 2
    theta = np.sqrt(rng.random(n_per)) * 3.5 * np.pi
    r = theta
    x1 = np.c_[r * np.cos(theta), r * np.sin(theta)]
    x2 = np.c_[-r * np.cos(theta), -r * np.sin(theta)]
    X = np.vstack([x1, x2])
    y = np.array([0] * n_per + [1] * n_per)
    X += rng.normal(scale=noise, size=X.shape)
    # normalize
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y


def build_datasets():
    datasets = []
    rng = np.random.default_rng(0)

    for n in [100, 200, 400]:
        X, y = make_moons(n_samples=n, noise=0.2, random_state=42)
        datasets.append((f"moons_{n}", X, y))

        X, y = make_circles(n_samples=n, noise=0.15, factor=0.5, random_state=42)
        datasets.append((f"circles_{n}", X, y))

        X, y = make_xor(n=n, noise=0.15, random_state=42)
        datasets.append((f"xor_{n}", X, y))

        X, y = make_spirals(n=n, noise=0.1, random_state=42)
        datasets.append((f"spirals_{n}", X, y))

    return datasets


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_models():
    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500)),
        ]),
        "MLP_32": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(32,), max_iter=2000, random_state=42
            )),
        ]),
        "MLP_100": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(100,), max_iter=2000, random_state=42
            )),
        ]),
        "KNN_k5": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "KNN_k10": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=10)),
        ]),
        "RF_100": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "SmallMLP_nw": SmallMLPClassifier(
            point_method="nw",
            class_weight="balanced",
            h_min=0.01, h_max=10.0,
            max_iter=30, inner_iter=10, tol=1e-8,
        ),
        "SmallMLP_klr": SmallMLPClassifier(
            point_method="klr",
            class_weight="balanced",
            lam=1e-3,
            h_min=0.01, h_max=10.0,
            max_iter=30, inner_iter=20, tol=1e-8,
        ),
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(n_splits=5):
    datasets = build_datasets()
    models = build_models()

    print(f"Datasets: {len(datasets)}")
    print(f"Models:   {len(models)}")
    print("=" * 100)

    rows = []
    for ds_name, X, y in datasets:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs = {name: [] for name in models}
        accs = {name: [] for name in models}

        for tr, te in skf.split(X, y):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]

            for name, model in models.items():
                try:
                    m = clone(model)
                    m.fit(X_tr, y_tr)
                    if hasattr(m, "predict_proba"):
                        p = m.predict_proba(X_te)[:, 1]
                    else:
                        p = m.decision_function(X_te)
                    aucs[name].append(roc_auc_score(y_te, p))
                    accs[name].append(accuracy_score(y_te, m.predict(X_te)))
                except Exception:
                    aucs[name].append(np.nan)
                    accs[name].append(np.nan)

        for name in models:
            rows.append({
                "dataset": ds_name,
                "model": name,
                "auc": np.nanmean(aucs[name]),
                "acc": np.nanmean(accs[name]),
                "n": len(y),
                "d": X.shape[1],
            })

        best = max(models, key=lambda nm: np.nanmean(aucs[nm]))
        print(f"{ds_name:16s}  best={best:14s}  "
              f"nw_AUC={np.nanmean(aucs['SmallMLP_nw']):.4f}  "
              f"klr_AUC={np.nanmean(aucs['SmallMLP_klr']):.4f}")

    return pd.DataFrame(rows)


def summarize(df):
    print("\n" + "=" * 100)
    print("SUMMARY: nonlinear classification")
    print("=" * 100)

    pivot_auc = df.pivot(index="dataset", columns="model", values="auc")

    print("\nMean AUC:")
    for name, m in pivot_auc.mean(axis=0).sort_values(ascending=False).items():
        print(f"  {name:14s}: {m:.4f}")

    print("\nMean rank (AUC, 1 = best):")
    ranks = pivot_auc.rank(axis=1, ascending=False, method="average")
    for name, r in ranks.mean(axis=0).sort_values().items():
        print(f"  {name:14s}: {r:.3f}")

    print("\nWin count (AUC):")
    wins = (pivot_auc == pivot_auc.max(axis=1).values[:, None]).sum(axis=0)
    for name, w in wins.sort_values(ascending=False).items():
        print(f"  {name:14s}: {int(w)}")

    # SmallMLP vs best
    comp_cols = [c for c in pivot_auc.columns
                 if c not in ("SmallMLP_nw", "SmallMLP_klr")]
    best_comp = pivot_auc[comp_cols].max(axis=1)
    print(f"\nSmallMLP_klr vs best competitor:")
    gap = pivot_auc["SmallMLP_klr"] - best_comp
    for ds_name, g in gap.items():
        marker = " +" if g > 0 else ""
        print(f"  {ds_name:16s}: {g:+.4f}{marker}")
    print(f"  mean gap: {gap.mean():+.4f}")


if __name__ == "__main__":
    df = run_benchmark(n_splits=5)
    df.to_csv("benchmarks/results_nonlinear_classification.csv", index=False)
    summarize(df)
    print("\nSaved: benchmarks/results_nonlinear_classification.csv")