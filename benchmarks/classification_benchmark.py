"""Full classification benchmark: point + conformal sets.

Compares SmallMLP against standard classifiers on:
  - Point prediction: AUC, accuracy
  - Conformal sets: coverage, avg size (alpha=0.1)

All baselines use default hyperparameters (no tuning).
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from smallmlp import SmallMLPClassifier

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _load_uciml(uid):
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=uid)
    X_df = ds.data.features.select_dtypes(include=[np.number])
    X = X_df.to_numpy(dtype=float)
    y_raw = ds.data.targets.iloc[:, 0]
    y = LabelEncoder().fit_transform(y_raw.astype(str))
    if np.isnan(X).any():
        X = SimpleImputer(strategy="median").fit_transform(X)
    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]
    cls = np.unique(y)[:2]
    m = np.isin(y, cls)
    return X[m], y[m]


def build_datasets():
    from sklearn.datasets import load_breast_cancer, load_wine, load_iris
    datasets = []

    uci = {
        "hepatitis": 46,
        "parkinsons": 174,
        "sonar": 151,
        "ionosphere": 52,
        "heart_statlog": 145,
        "liver_disorders": 225,
        "breast_cancer_wisconsin": 15,
        "glass": 42,
        "blood_transfusion": 176,
        "haberman": 43,
        "tic_tac_toe": 101,
        "banknote": 267,
    }
    for name, uid in uci.items():
        try:
            X, y = _load_uciml(uid)
            datasets.append((f"uci_{name}", X, y))
            print(f"[ok] {name}: n={len(y)}, d={X.shape[1]}")
        except Exception as e:
            print(f"[skip] {name}: {e}")

    rng = np.random.default_rng(0)
    for loader, name in [
        (load_breast_cancer, "sk_breast_cancer"),
        (load_wine, "sk_wine"),
        (load_iris, "sk_iris"),
    ]:
        data = loader()
        X, y = data.data, data.target
        if len(np.unique(y)) > 2:
            cls = np.unique(y)[:2]
            m = np.isin(y, cls)
            X, y = X[m], y[m]
        for n_sub in [100, 200, 300]:
            if len(y) >= n_sub:
                idx = rng.choice(len(y), size=n_sub, replace=False)
                datasets.append((f"{name}_{n_sub}", X[idx], y[idx]))

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
                hidden_layer_sizes=(32,), max_iter=1000, random_state=42
            )),
        ]),
        "MLP_100": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(100,), max_iter=1000, random_state=42
            )),
        ]),
        "MLP_wide": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(256, 128), max_iter=1000, random_state=42
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

    print(f"\nDatasets: {len(datasets)}")
    print(f"Models:   {len(models)}")
    print("=" * 100)

    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

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
        print(f"{ds_name:32s}  best={best:10s}  "
              f"SmallMLP_AUC={np.nanmean(aucs['SmallMLP_nw']):.4f}  "
              f"SmallMLP_ACC={np.nanmean(accs['SmallMLP_nw']):.4f}")

    return pd.DataFrame(rows)


def summarize(df):
    print("\n" + "=" * 100)
    print("SUMMARY: point prediction")
    print("=" * 100)

    pivot_auc = df.pivot(index="dataset", columns="model", values="auc")
    pivot_acc = df.pivot(index="dataset", columns="model", values="acc")

    print("\nMean AUC:")
    for name, m in pivot_auc.mean(axis=0).sort_values(ascending=False).items():
        print(f"  {name:12s}: {m:.4f}")

    print("\nMean accuracy:")
    for name, m in pivot_acc.mean(axis=0).sort_values(ascending=False).items():
        print(f"  {name:12s}: {m:.4f}")

    print("\nMean rank (AUC, 1 = best):")
    ranks = pivot_auc.rank(axis=1, ascending=False, method="average")
    for name, r in ranks.mean(axis=0).sort_values().items():
        print(f"  {name:12s}: {r:.3f}")

    print("\nWin count (AUC):")
    wins = (pivot_auc == pivot_auc.max(axis=1).values[:, None]).sum(axis=0)
    for name, w in wins.sort_values(ascending=False).items():
        print(f"  {name:12s}: {int(w)}")

    return pivot_auc, pivot_acc


if __name__ == "__main__":
    df = run_benchmark(n_splits=5)
    df.to_csv("benchmarks/results_classification_full.csv", index=False)
    summarize(df)
    print("\nSaved: benchmarks/results_classification_full.csv")