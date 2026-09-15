"""Test hypothesis: SmallMLP wins when d/n is high.

Vary feature count d and sample size n on real scientific datasets.
Plot (or print) where SmallMLP beats competitors.
"""

import warnings
import numpy as np
import pandas as pd
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
    # binarize: first two classes
    cls = np.unique(y)[:2]
    m = np.isin(y, cls)
    return X[m], y[m]


def build_models():
    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500)),
        ]),
        "MLP_100": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(100,), max_iter=1000, random_state=42
            )),
        ]),
        "KNN_k5": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "RF_100": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "SmallMLP": SmallMLPClassifier(
            h_min=0.01, h_max=10.0,
            max_iter=30, inner_iter=10, tol=1e-8, verbose=False,
        ),
    }


def evaluate(X, y, models, n_splits=5):
    """Return mean AUC per model over CV splits."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {name: [] for name in models}
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
                results[name].append(roc_auc_score(y_te, p))
            except Exception:
                results[name].append(np.nan)
    return {k: np.nanmean(v) for k, v in results.items()}


def subsample(X, y, n, rng):
    if n >= len(y):
        return X, y
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], y[idx]


def run_dim_scaling(name, uid):
    print(f"\n{'='*70}")
    print(f"Dataset: {name}  (uid={uid})")
    print(f"{'='*70}")

    X_full, y_full = _load_uciml(uid)
    d_full = X_full.shape[1]
    n_full = len(y_full)
    print(f"Full: n={n_full}, d={d_full}\n")

    models = build_models()
    rng = np.random.default_rng(0)

    # --- 1. Vary d (feature count), full n ---
    print(f"{'d':>5} {'n':>5} {'d/n':>6} | " + " | ".join(f"{k:>10}" for k in models))
    print("-" * 100)

    rows = []
    d_values = sorted(set([5, 10, min(20, d_full), min(30, d_full), d_full]))
    d_values = [d for d in d_values if d <= d_full]

    for d in d_values:
        # take first d features (or a random subset for fairness)
        feat_idx = np.arange(d)
        X = X_full[:, feat_idx]
        y = y_full
        res = evaluate(X, y, models)
        d_over_n = d / len(y)
        row = {"dataset": name, "d": d, "n": len(y), "d_over_n": d_over_n, **res}
        rows.append(row)
        line = f"{d:>5} {len(y):>5} {d_over_n:>6.3f} | "
        line += " | ".join(f"{res[k]:>10.4f}" for k in models)
        print(line)

    # --- 2. Vary n (sample size), full d ---
    print()
    print(f"{'d':>5} {'n':>5} {'d/n':>6} | " + " | ".join(f"{k:>10}" for k in models))
    print("-" * 100)

    n_values = sorted(set([50, 100, 200, n_full]))
    n_values = [n for n in n_values if n <= n_full]

    for n in n_values:
        X_sub, y_sub = subsample(X_full, y_full, n, rng)
        res = evaluate(X_sub, y_sub, models)
        d = d_full
        d_over_n = d / n
        row = {"dataset": name, "d": d, "n": n, "d_over_n": d_over_n, **res}
        rows.append(row)
        line = f"{d:>5} {n:>5} {d_over_n:>6.3f} | "
        line += " | ".join(f"{res[k]:>10.4f}" for k in models)
        print(line)

    return pd.DataFrame(rows)


def main():
    datasets = {
        "ionosphere": 52,
        "sonar": 151,
        "parkinsons": 174,
    }

    all_rows = []
    for name, uid in datasets.items():
        df = run_dim_scaling(name, uid)
        all_rows.append(df)

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv("benchmarks/dim_scaling.csv", index=False)
    print("\nSaved: benchmarks/dim_scaling.csv")

    # --- Summary: where does SmallMLP win? ---
    print("\n" + "=" * 70)
    print("SmallMLP vs best competitor, as function of d/n")
    print("=" * 70)

    competitor_cols = [c for c in result.columns
                       if c not in ("dataset", "d", "n", "d_over_n", "SmallMLP")]
    result["best_competitor"] = result[competitor_cols].max(axis=1)
    result["smallmlp_gap"] = result["SmallMLP"] - result["best_competitor"]

    print(f"\n{'dataset':<14} {'d':>4} {'n':>5} {'d/n':>7} {'SmallMLP':>10} "
          f"{'best_comp':>11} {'gap':>8}")
    print("-" * 70)
    for _, r in result.iterrows():
        marker = " +" if r["smallmlp_gap"] > 0 else ""
        print(f"{r['dataset']:<14} {int(r['d']):>4} {int(r['n']):>5} "
              f"{r['d_over_n']:>7.3f} {r['SmallMLP']:>10.4f} "
              f"{r['best_competitor']:>11.4f} {r['smallmlp_gap']:>+8.4f}{marker}")

    # correlation
    corr = result["d_over_n"].corr(result["smallmlp_gap"])
    print(f"\nCorrelation(d/n, SmallMLP gap): {corr:+.3f}")


if __name__ == "__main__":
    main()