"""Benchmark: SmallMLPClassifier vs standard classifiers on small
real-world scientific datasets (n < 500).

All baselines use default-ish hyperparameters (no tuning).
Metric: ROC-AUC, 5-fold CV.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.impute import SimpleImputer

from smallmlp import SmallMLPClassifier

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _load_uciml(id_or_name):
    from ucimlrepo import fetch_ucirepo
    if isinstance(id_or_name, int):
        ds = fetch_ucirepo(id=id_or_name)
    else:
        ds = fetch_ucirepo(name=id_or_name)

    X_df = ds.data.features
    y_df = ds.data.targets

    # numeric features only; impute NaN
    X_df = X_df.select_dtypes(include=[np.number])
    X = X_df.to_numpy(dtype=float)

    # target: first column
    if hasattr(y_df, "columns"):
        y_raw = y_df.iloc[:, 0]
    else:
        y_raw = y_df
    y = LabelEncoder().fit_transform(y_raw.astype(str))

    # impute NaN in features (median) — SmallMLP doesn't handle NaN
    if np.isnan(X).any():
        imp = SimpleImputer(strategy="median")
        X = imp.fit_transform(X)

    # keep only finite rows
    mask = np.isfinite(X).all(axis=1)
    return X[mask], y[mask]


def build_datasets():
    datasets = []

    # UCI small scientific datasets
    uci = {
        "hepatitis": 46,
        "parkinsons": 174,
        "sonar": 151,
        "ionosphere": 52,
        "heart_statlog": 145,
        "liver_disorders": 225,
        "breast_cancer_wisconsin": 15,
        "glass": 42,
    }
    for name, uid in uci.items():
        try:
            X, y = _load_uciml(uid)
            if len(np.unique(y)) == 2:
                datasets.append((f"uci_{name}", X, y))
            else:
                # one-vs-rest for multiclass: take first two classes only
                cls = np.unique(y)[:2]
                mask = np.isin(y, cls)
                datasets.append((f"uci_{name}_2cls", X[mask], y[mask]))
            print(f"[ok] {name}: n={len(y)}, d={X.shape[1]}")
        except Exception as e:
            print(f"[skip] {name}: {e}")

    # sklearn built-in (subsampled to small-data regime)
    rng = np.random.default_rng(0)
    for loader, name in [
        (load_breast_cancer, "sk_breast_cancer"),
        (load_wine, "sk_wine"),
        (load_iris, "sk_iris"),
    ]:
        data = loader()
        X, y = data.data, data.target
        # binarize multiclass
        if len(np.unique(y)) > 2:
            cls = np.unique(y)[:2]
            mask = np.isin(y, cls)
            X, y = X[mask], y[mask]
        for n_sub in [100, 200]:
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
                hidden_layer_sizes=(32,),
                max_iter=1000,
                random_state=42,
            )),
        ]),
        "MLP_100": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=1000,
                random_state=42,
            )),
        ]),
        "MLP_wide": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=1000,
                random_state=42,
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
        "SmallMLP": SmallMLPClassifier(
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
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = {name: [] for name in model_names}

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
                    fold_results[name].append(roc_auc_score(y_te, p))
                except Exception:
                    fold_results[name].append(np.nan)

        for name in model_names:
            rows.append({
                "dataset": ds_name,
                "model": name,
                "auc": np.nanmean(fold_results[name]),
            })

        best = max(
            model_names,
            key=lambda nm: np.nanmean(fold_results[nm])
        )
        print(f"{ds_name:32s}  best={best:10s}  "
              f"SmallMLP_AUC={np.nanmean(fold_results['SmallMLP']):.4f}")

    return pd.DataFrame(rows)


def summarize(df):
    print("\n" + "=" * 70)
    print("SUMMARY: mean ROC-AUC across datasets (higher is better)")
    print("=" * 70)

    pivot = df.pivot(index="dataset", columns="model", values="auc")

    print("\nMean AUC:")
    for name, a in pivot.mean(axis=0).sort_values(ascending=False).items():
        print(f"  {name:12s}: {a:.4f}")

    print("\nMean rank (1 = best):")
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    for name, r in ranks.mean(axis=0).sort_values().items():
        print(f"  {name:12s}: {r:.3f}")

    print("\nWin count (best AUC per dataset):")
    wins = (pivot == pivot.max(axis=1).values[:, None]).sum(axis=0).sort_values(
        ascending=False
    )
    for name, w in wins.items():
        print(f"  {name:12s}: {int(w)}")

    return pivot


if __name__ == "__main__":
    df = run_benchmark(n_splits=5)
    df.to_csv("benchmarks/results_classifier.csv", index=False)
    pivot = summarize(df)
    pivot.to_csv("benchmarks/pivot_classifier.csv")
    print("\nSaved: benchmarks/results_classifier.csv, benchmarks/pivot_classifier.csv")