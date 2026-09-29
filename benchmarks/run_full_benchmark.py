"""Full benchmark: SmallMLP vs baselines.

Regression: MAE, 5-fold CV.
Classification: accuracy, 5-fold CV.

Conformal: coverage, width/set size (alpha=0.1), 60/20/20 split.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.datasets import (
    load_diabetes, load_breast_cancer, load_wine, load_iris,
    make_friedman1, make_friedman2, make_friedman3, make_regression,
    fetch_openml,
)

from smallmlp import SmallMLPRegressor, SmallMLPClassifier

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Regression datasets
# ---------------------------------------------------------------------------

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


def build_regression_datasets():
    datasets = []

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

    for n in [100, 300, 500]:
        for name, fn in [("friedman1", make_friedman1),
                         ("friedman2", make_friedman2),
                         ("friedman3", make_friedman3)]:
            try:
                X, y = fn(n_samples=n, noise=0.1, random_state=42)
                datasets.append((f"{name}_{n}", X, y))
            except Exception:
                pass

    for n, d, n_inf in [(100, 20, 5), (300, 30, 8), (500, 40, 10)]:
        X, y = make_regression(
            n_samples=n, n_features=d, n_informative=n_inf,
            noise=0.3, random_state=42,
        )
        datasets.append((f"mreg_{n}x{d}_inf{n_inf}", X, y))

    diab = load_diabetes()
    datasets.append(("diabetes", diab.data, diab.target))
    rng = np.random.default_rng(7)
    for n_sub in [50, 100, 200]:
        idx = rng.choice(len(diab.target), size=n_sub, replace=False)
        datasets.append((f"diabetes_{n_sub}", diab.data[idx], diab.target[idx]))

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
            for n_sub in [100, 300, 500]:
                if len(y) >= n_sub:
                    idx = rng.choice(len(y), size=n_sub, replace=False)
                    datasets.append((f"{name}_{n_sub}", X[idx], y[idx]))
        except Exception as e:
            print(f"[skip] {name}: {e}")

    return datasets


# ---------------------------------------------------------------------------
# Classification datasets
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
    return X[mask], y[mask]


def subsample(X, y, n_max=500, min_per_class=5, seed=0):
    """Subsample to n_max, ensuring >= min_per_class per class.

    For multiclass with K>8, raises n_max to max(n_max, min_per_class * K).
    Skips dataset if not enough samples to satisfy min_per_class.
    """
    K = len(np.unique(y))
    n_max_eff = max(n_max, min_per_class * K) if K > 8 else n_max
    if len(y) <= n_max_eff:
        counts = np.bincount(y)
        if counts.min() < min_per_class:
            return None
        return X, y
    rng = np.random.default_rng(seed)
    # stratified subsample to preserve class balance
    idx = []
    for k in np.unique(y):
        k_idx = np.where(y == k)[0]
        n_k = max(min_per_class, int(round(n_max_eff * len(k_idx) / len(y))))
        n_k = min(n_k, len(k_idx))
        idx.extend(rng.choice(k_idx, size=n_k, replace=False))
    idx = np.array(idx)
    return X[idx], y[idx]


def build_classification_datasets():
    datasets = []

    for loader, name in [
        (load_iris, "iris"),
        (load_wine, "wine"),
        (load_breast_cancer, "breast_cancer"),
    ]:
        data = loader()
        r = subsample(data.data, data.target, n_max=500)
        if r is not None:
            datasets.append((f"sk_{name}", r[0], r[1]))

    uci = {
        "hepatitis": 46, "parkinsons": 174, "sonar": 151,
        "ionosphere": 52, "heart_statlog": 145, "liver_disorders": 225,
        "blood_transfusion": 176,
        "banknote": 267,
    }
    for name, uid in uci.items():
        try:
            X, y = _load_uciml(uid)
            r = subsample(X, y, n_max=500)
            if r is not None:
                datasets.append((f"uci_{name}", r[0], r[1]))
        except Exception as e:
            print(f"[skip] {name}: {e}")

    uci_multi = {
        "glass": 42,
        "yeast": 110,
        "vowel": 59,
        "segment": 50,
        "cmc": 30,
        "balance_scale": 12,
        "wine_quality_red": 186,
        "wine_quality_white": 187,
    }
    for name, uid in uci_multi.items():
        try:
            X, y = _load_uciml(uid)
            r = subsample(X, y, n_max=500, min_per_class=5)
            if r is not None:
                datasets.append((f"uci_{name}", r[0], r[1]))
        except Exception as e:
            print(f"[skip] {name}: {e}")

    return datasets


# ---------------------------------------------------------------------------
# Regression benchmark
# ---------------------------------------------------------------------------

def run_regression_benchmark(n_splits=5):
    from scipy.stats import wilcoxon

    datasets = build_regression_datasets()

    models = {
        "MLP_100": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(100,),
                                    max_iter=1000, random_state=42)),
        ]),
        "MLP_wide": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(256, 128),
                                    max_iter=1000, random_state=42)),
        ]),
        "KNN_k5": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=5)),
        ]),
        "RF_100": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "SmallMLP": SmallMLPRegressor(max_epochs=500, patience=30),
    }

    print(f"\nREGRESSION: {len(datasets)} datasets, {len(models)} models")
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
            rows.append({"dataset": ds_name, "model": name,
                         "mae": np.nanmean(maes[name])})
        best = min(models, key=lambda nm: np.nanmean(maes[nm]))
        print(f"{ds_name:28s}  best={best:12s}  "
              f"SmallMLP_MAE={np.nanmean(maes['SmallMLP']):.4f}")

    df = pd.DataFrame(rows)

    pivot = df.pivot(index="dataset", columns="model", values="mae")
    ranks = pivot.rank(axis=1, method="average")
    mean_rank = ranks.mean(axis=0).sort_values()

    print("\n" + "=" * 100)
    print("Mean rank (lower = better) and mean MAE:")
    print(f"{'model':<12} {'mean_rank':>10} {'mean_mae':>10}")
    for name in mean_rank.index:
        print(f"{name:<12} {mean_rank[name]:>10.3f} {pivot[name].mean():>10.4f}")

    print("\n" + "=" * 100)
    print("Wilcoxon signed-rank test (SmallMLP vs baseline, lower MAE = better):")
    print(f"{'baseline':<12} {'statistic':>12} {'p-value':>12} {'wins':>6} {'losses':>7}")
    sm = pivot["SmallMLP"].values
    for name in pivot.columns:
        if name == "SmallMLP":
            continue
        other = pivot[name].values
        wins = int(np.sum(sm < other - 1e-9))
        losses = int(np.sum(sm > other + 1e-9))
        try:
            stat, p = wilcoxon(sm, other, zero_method="wilcox", alternative="two-sided")
        except Exception:
            stat, p = np.nan, np.nan
        print(f"{name:<12} {stat:>12.3f} {p:>12.4f} {wins:>6} {losses:>7}")

    return df


# ---------------------------------------------------------------------------
# Classification benchmark
# ---------------------------------------------------------------------------

def _bucket(K):
    if K == 2:
        return "binary"
    if K <= 5:
        return "multiclass_2_5"
    return "multiclass_gt5"


def run_classification_benchmark(n_splits=5):
    from scipy.stats import wilcoxon

    datasets = build_classification_datasets()

    models = {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500)),
        ]),
        "MLP_100": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(100,),
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
        "SmallMLP": SmallMLPClassifier(class_weight="balanced",
                                        max_epochs=500, patience=30),
    }

    print(f"\n\nCLASSIFICATION: {len(datasets)} datasets, {len(models)} models")
    print("=" * 100)

    rows = []
    per_ds_acc = {}

    for ds_name, X, y in datasets:
        K = len(np.unique(y))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs = {name: [] for name in models}
        for tr, te in skf.split(X, y):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            for name, model in models.items():
                try:
                    m = clone(model)
                    m.fit(X_tr, y_tr)
                    accs[name].append(accuracy_score(y_te, m.predict(X_te)))
                except Exception:
                    accs[name].append(np.nan)

        ds_acc = {name: float(np.nanmean(accs[name])) for name in models}
        per_ds_acc[ds_name] = ds_acc

        for name in models:
            rows.append({"dataset": ds_name, "model": name,
                         "acc": ds_acc[name], "K": K, "bucket": _bucket(K)})

        best = max(models, key=lambda nm: ds_acc[nm])
        print(f"{ds_name:28s}  K={K:>3}  best={best:12s}  "
              f"SmallMLP_acc={ds_acc['SmallMLP']:.4f}")

    df = pd.DataFrame(rows)

    # ---- overall mean rank ----
    pivot = df.pivot(index="dataset", columns="model", values="acc")
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    mean_rank = ranks.mean(axis=0).sort_values()
    mean_acc = pivot.mean(axis=0).sort_values(ascending=False)

    print("\n" + "=" * 100)
    print("Overall — mean rank (lower = better) and mean accuracy:")
    print(f"{'model':<12} {'mean_rank':>10} {'mean_acc':>10}")
    for name in mean_rank.index:
        print(f"{name:<12} {mean_rank[name]:>10.3f} {mean_acc[name]:>10.4f}")

    # ---- per-bucket ----
    print("\n" + "=" * 100)
    print("Per-K-bucket mean accuracy:")
    buckets = ["binary", "multiclass_2_5", "multiclass_gt5"]
    print(f"{'bucket':<18} {'n':>3} " +
          " ".join(f"{m:>10}" for m in mean_acc.index))
    for b in buckets:
        sub = df[df["bucket"] == b]
        if len(sub) == 0:
            continue
        sub_pivot = sub.pivot(index="dataset", columns="model", values="acc")
        n_ds = sub_pivot.shape[0]
        row = f"{b:<18} {n_ds:>3}"
        for name in mean_acc.index:
            row += f" {sub_pivot[name].mean():>10.4f}"
        print(row)

    # ---- Wilcoxon ----
    print("\n" + "=" * 100)
    print("Wilcoxon signed-rank test (SmallMLP vs baseline):")
    print(f"{'baseline':<12} {'statistic':>12} {'p-value':>12} {'wins':>6} {'losses':>7}")
    sm = pivot["SmallMLP"].values
    for name in pivot.columns:
        if name == "SmallMLP":
            continue
        other = pivot[name].values
        wins = int(np.sum(sm > other + 1e-9))
        losses = int(np.sum(sm < other - 1e-9))
        try:
            stat, p = wilcoxon(sm, other, zero_method="wilcox", alternative="two-sided")
        except Exception:
            stat, p = np.nan, np.nan
        print(f"{name:<12} {stat:>12.3f} {p:>12.4f} {wins:>6} {losses:>7}")

    return df


# ---------------------------------------------------------------------------
# Conformal evaluation
# ---------------------------------------------------------------------------

def run_conformal_eval(alpha=0.1):
    print("\n\nCONFORMAL EVALUATION (alpha=0.1)")
    print("=" * 100)

    print("\n--- Regression ---")
    print(f"{'dataset':<28} {'n_tr':>5} {'n_cal':>6} {'n_val':>6} "
          f"{'coverage':>10} {'width':>10}")
    print("-" * 100)

    reg_rows = []
    for ds_name, X, y in build_regression_datasets():
        try:
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                X, y, test_size=0.4, random_state=0
            )
            X_cal, X_val, y_cal, y_val = train_test_split(
                X_tmp, y_tmp, test_size=0.5, random_state=0
            )
            reg = SmallMLPRegressor(max_epochs=500, patience=30)
            reg.fit(X_tr, y_tr)
            reg.fit_conformal(X_cal, y_cal, X_val, y_val, alpha=alpha)
            lo, hi = reg.predict_interval_conformal(X_val, alpha=alpha)
            cov = float(np.mean((y_val >= lo) & (y_val <= hi)))
            width = float(np.mean(hi - lo))
            reg_rows.append({"dataset": ds_name, "coverage": cov, "width": width})
            print(f"{ds_name:<28} {len(y_tr):>5} {len(y_cal):>6} {len(y_val):>6} "
                  f"{cov:>10.3f} {width:>10.3f}")
        except Exception as e:
            print(f"{ds_name:<28} FAILED: {e}")

    print("\n--- Classification ---")
    print(f"{'dataset':<28} {'K':>3} {'n_tr':>5} {'n_cal':>6} {'n_val':>6} "
          f"{'coverage':>10} {'size':>10}")
    print("-" * 100)

    clf_rows = []
    for ds_name, X, y in build_classification_datasets():
        try:
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                X, y, test_size=0.4, random_state=0, stratify=y
            )
            X_cal, X_val, y_cal, y_val = train_test_split(
                X_tmp, y_tmp, test_size=0.5, random_state=0, stratify=y_tmp
            )
            clf = SmallMLPClassifier(class_weight="balanced",
                                      max_epochs=500, patience=30)
            clf.fit(X_tr, y_tr)
            clf.fit_conformal(X_cal, y_cal, X_val, y_val, alpha=alpha)
            sets = clf.predict_set(X_val, alpha=alpha)
            y_enc = clf._label_encoder.transform(y_val)
            cov = float(np.mean([y_enc[i] in sets[i] for i in range(len(y_val))]))
            size = float(np.mean([len(s) for s in sets]))
            clf_rows.append({"dataset": ds_name, "coverage": cov, "size": size,
                             "K": clf.n_classes_})
            print(f"{ds_name:<28} {clf.n_classes_:>3} {len(y_tr):>5} "
                  f"{len(y_cal):>6} {len(y_val):>6} {cov:>10.3f} {size:>10.3f}")
        except Exception as e:
            print(f"{ds_name:<28} FAILED: {e}")

    return pd.DataFrame(reg_rows), pd.DataFrame(clf_rows)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize_regression(df):
    print("\n" + "=" * 100)
    print("REGRESSION SUMMARY")
    print("=" * 100)
    pivot = df.pivot(index="dataset", columns="model", values="mae")
    print("\nMean MAE:")
    for name, m in pivot.mean(axis=0).sort_values().items():
        print(f"  {name:12s}: {m:.4f}")
    print("\nMean rank:")
    for name, r in pivot.rank(axis=1, method="average").mean(axis=0).sort_values().items():
        print(f"  {name:12s}: {r:.3f}")
    print("\nWin count:")
    wins = (pivot == pivot.min(axis=1).values[:, None]).sum(axis=0)
    for name, w in wins.sort_values(ascending=False).items():
        print(f"  {name:12s}: {int(w)}")


def summarize_classification(df):
    print("\n" + "=" * 100)
    print("CLASSIFICATION SUMMARY")
    print("=" * 100)
    pivot = df.pivot(index="dataset", columns="model", values="acc")
    print("\nMean accuracy:")
    for name, m in pivot.mean(axis=0).sort_values(ascending=False).items():
        print(f"  {name:12s}: {m:.4f}")
    print("\nMean rank:")
    for name, r in pivot.rank(axis=1, ascending=False, method="average").mean(axis=0).sort_values().items():
        print(f"  {name:12s}: {r:.3f}")
    print("\nWin count:")
    wins = (pivot == pivot.max(axis=1).values[:, None]).sum(axis=0)
    for name, w in wins.sort_values(ascending=False).items():
        print(f"  {name:12s}: {int(w)}")

    if "bucket" in df.columns:
        print("\nPer-bucket mean accuracy:")
        for b in ["binary", "multiclass_2_5", "multiclass_gt5"]:
            sub = df[df["bucket"] == b]
            if len(sub) == 0:
                continue
            sub_pivot = sub.pivot(index="dataset", columns="model", values="acc")
            print(f"\n  [{b}]  n_datasets={sub_pivot.shape[0]}")
            for name, m in sub_pivot.mean(axis=0).sort_values(ascending=False).items():
                print(f"    {name:12s}: {m:.4f}")


if __name__ == "__main__":
    # reg_df = run_regression_benchmark(n_splits=5)
    # reg_df.to_csv("benchmarks/results_regression.csv", index=False)
    # summarize_regression(reg_df)

    clf_df = run_classification_benchmark(n_splits=5)
    clf_df.to_csv("benchmarks/results_classification.csv", index=False)
    summarize_classification(clf_df)

    # reg_conf, clf_conf = run_conformal_eval(alpha=0.1)
    # reg_conf.to_csv("benchmarks/results_conformal_regression.csv", index=False)
    # clf_conf.to_csv("benchmarks/results_conformal_classification.csv", index=False)

    print("\nSaved 4 CSV files to benchmarks/")