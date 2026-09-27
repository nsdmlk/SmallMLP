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


def subsample(X, y, n_max=500, seed=0):
    if len(y) <= n_max:
        return X, y
    K = len(np.unique(y))
    if K > 8:
        n_max = max(n_max, 1000)
    idx = np.random.default_rng(seed).choice(len(y), size=n_max, replace=False)
    return X[idx], y[idx]


def build_classification_datasets():
    datasets = []

    for loader, name in [
        (load_iris, "iris"),
        (load_wine, "wine"),
        (load_breast_cancer, "breast_cancer"),
    ]:
        data = loader()
        X, y = subsample(data.data, data.target, n_max=500)
        datasets.append((f"sk_{name}", X, y))

    uci = {
        "hepatitis": 46, "parkinsons": 174, "sonar": 151,
        "ionosphere": 52, "heart_statlog": 145, "liver_disorders": 225,
        "blood_transfusion": 176,
        "banknote": 267,
    }
    for name, uid in uci.items():
        try:
            X, y = _load_uciml(uid)
            X, y = subsample(X, y, n_max=500)
            datasets.append((f"uci_{name}", X, y))
        except Exception as e:
            print(f"[skip] {name}: {e}")

    uci_multi = {
        "glass": 42,
        "yeast": 110,
        "vowel": 59,
        "vehicle": 149,
    }
    for name, uid in uci_multi.items():
        try:
            X, y = _load_uciml(uid)
            X, y = subsample(X, y, n_max=500)
            datasets.append((f"uci_{name}", X, y))
        except Exception as e:
            print(f"[skip] {name}: {e}")

    return datasets


# ---------------------------------------------------------------------------
# Regression benchmark
# ---------------------------------------------------------------------------

def run_regression_benchmark(n_splits=5):
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

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Classification benchmark
# ---------------------------------------------------------------------------

def run_classification_benchmark(n_splits=5):
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
    for ds_name, X, y in datasets:
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
        for name in models:
            rows.append({"dataset": ds_name, "model": name,
                         "acc": np.nanmean(accs[name])})
        best = max(models, key=lambda nm: np.nanmean(accs[nm]))
        print(f"{ds_name:28s}  best={best:12s}  "
              f"SmallMLP_acc={np.nanmean(accs['SmallMLP']):.4f}")

    return pd.DataFrame(rows)


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


if __name__ == "__main__":
    reg_df = run_regression_benchmark(n_splits=5)
    reg_df.to_csv("benchmarks/results_regression.csv", index=False)
    summarize_regression(reg_df)

    clf_df = run_classification_benchmark(n_splits=5)
    clf_df.to_csv("benchmarks/results_classification.csv", index=False)
    summarize_classification(clf_df)

    reg_conf, clf_conf = run_conformal_eval(alpha=0.1)
    reg_conf.to_csv("benchmarks/results_conformal_regression.csv", index=False)
    clf_conf.to_csv("benchmarks/results_conformal_classification.csv", index=False)

    print("\nSaved 4 CSV files to benchmarks/")