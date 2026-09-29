"""
Full classification benchmark with extended metrics.

Metrics:
  - mean accuracy
  - median accuracy
  - mean rank (lower better)
  - win count (best on dataset)
  - top-3 count (in top 3 on dataset)
  - Wilcoxon vs SmallMLP

Run:
  python benchmark_classification_full.py
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from smallmlp import SmallMLPClassifier
from run_full_benchmark import build_classification_datasets

warnings.filterwarnings("ignore")


def make_models():
    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500)),
        ]),
        "MLP_100": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(100,),
                                     max_iter=1000, random_state=42)),
        ]),
        "MLP_128": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(128,),
                                     max_iter=1000, random_state=42)),
        ]),
        "MLP_100_100": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(100, 100),
                                     max_iter=1000, random_state=42)),
        ]),
        "KNN_k5": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "RF_100": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVC_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale",
                          probability=True, random_state=42)),
        ]),
        "SmallMLP_formula": SmallMLPClassifier(
            width_mode="formula", class_weight=None,
            alpha=4.0, beta=0.7, bias_init="kaiming",
            max_epochs=500, patience=30, random_state=42,
        ),
        "SmallMLP_classic": SmallMLPClassifier(
            width_mode="classic", class_weight=None,
            max_epochs=500, patience=30, random_state=42,
        ),
        "SmallMLP_bal": SmallMLPClassifier(
            width_mode="formula", class_weight="balanced",
            alpha=4.0, beta=0.7, max_epochs=500, patience=30, random_state=42,
        ),
    }


def run_benchmark(datasets, n_splits=5, seed=42):
    models = make_models()
    rows = []
    for name, X, y in datasets:
        K = len(np.unique(y))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        accs = {m: [] for m in models}
        for tr, te in skf.split(X, y):
            for mname, model in models.items():
                try:
                    m = clone(model)
                    m.fit(X[tr], y[tr])
                    acc = accuracy_score(y[te], m.predict(X[te]))
                except Exception as e:
                    warnings.warn(f"{name} ({mname}): {e}")
                    acc = np.nan
                accs[mname].append(acc)
        for mname in models:
            rows.append({
                "dataset": name, "K": K, "model": mname,
                "acc": float(np.nanmean(accs[mname])),
            })
    return pd.DataFrame(rows)


def summarize(df):
    pivot = df.pivot(index="dataset", columns="model", values="acc")
    names = pivot.index.tolist()
    models = pivot.columns.tolist()

    print("\n" + "=" * 110)
    print("MEAN ACCURACY")
    print("=" * 110)
    print(f"{'model':<20} {'mean':>10} {'median':>10} {'std':>10}")
    for m in models:
        vals = pivot[m].values
        print(f"{m:<20} {np.mean(vals):>10.4f} {np.median(vals):>10.4f} "
              f"{np.std(vals):>10.4f}")

    print("\n" + "=" * 110)
    print("MEAN RANK (lower = better)")
    print("=" * 110)
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    mean_rank = ranks.mean(axis=0).sort_values()
    for m in mean_rank.index:
        print(f"{m:<20} {mean_rank[m]:>10.3f}")

    print("\n" + "=" * 110)
    print("WIN COUNT (best on dataset)")
    print("=" * 110)
    wins = (pivot == pivot.max(axis=1).values[:, None]).sum(axis=0).sort_values(ascending=False)
    for m in wins.index:
        print(f"{m:<20} {int(wins[m]):>10}")

    print("\n" + "=" * 110)
    print("TOP-3 COUNT (in top 3 on dataset)")
    print("=" * 110)
    top3 = np.zeros(len(models))
    for i, ds in enumerate(names):
        row = pivot.loc[ds].values
        order = np.argsort(-row)
        for j in order[:3]:
            top3[j] += 1
    top3_series = pd.Series(top3, index=models).sort_values(ascending=False)
    for m in top3_series.index:
        print(f"{m:<20} {int(top3_series[m]):>10}")

    print("\n" + "=" * 110)
    print("PAIRWISE WINS (row > col)")
    print("=" * 110)
    print(f"{'':<20} " + " ".join(f"{m[:8]:>8}" for m in models))
    for a in models:
        row = f"{a:<20} "
        for b in models:
            if a == b:
                row += f"{'—':>8} "
            else:
                w = sum(1 for n in names if pivot.loc[n, a] > pivot.loc[n, b] + 1e-9)
                row += f"{w:>8} "
        print(row)

    print("\n" + "=" * 110)
    print("WILCOXON vs SmallMLP_formula (paired across datasets)")
    print("=" * 110)
    ref = "SmallMLP_formula"
    if ref in pivot.columns:
        sm = pivot[ref].values
        print(f"{'baseline':<20} {'stat':>10} {'p':>10} {'wins':>6} {'losses':>8}")
        for m in models:
            if m == ref:
                continue
            other = pivot[m].values
            wins = int(np.sum(sm > other + 1e-9))
            losses = int(np.sum(sm < other - 1e-9))
            try:
                stat, p = wilcoxon(sm, other, zero_method="wilcox", alternative="two-sided")
            except Exception:
                stat, p = np.nan, np.nan
            print(f"{m:<20} {stat:>10.3f} {p:>10.4f} {wins:>6} {losses:>8}")

    print("\n" + "=" * 110)
    print("PER-K-BUCKET mean accuracy")
    print("=" * 110)
    def bucket(K):
        if K == 2: return "binary"
        if K <= 5: return "multiclass_2_5"
        return "multiclass_gt5"
    Kmap = df.groupby("dataset")["K"].first().to_dict()
    bmap = {n: bucket(Kmap[n]) for n in names}
    for b in ["binary", "multiclass_2_5", "multiclass_gt5"]:
        subset = [n for n in names if bmap[n] == b]
        if not subset:
            continue
        print(f"\n  [{b}] n={len(subset)}")
        for m in models:
            vals = [pivot.loc[n, m] for n in subset]
            print(f"    {m:<20} {np.mean(vals):>10.4f}")


def main():
    datasets = build_classification_datasets()
    datasets = [(n, X, y) for n, X, y in datasets if np.bincount(y).min() >= 5]
    print(f"Using {len(datasets)} datasets")

    df = run_benchmark(datasets, n_splits=5, seed=42)
    df.to_csv("benchmarks/results_classification_full.csv", index=False)
    print("Saved to benchmarks/results_classification_full.csv")
    summarize(df)


if __name__ == "__main__":
    main()