"""Benchmark: weighted conformal classification vs split conformal.

Splits each dataset 60/20/20 (train / calibration / validation).
For alpha = 0.1, reports coverage, avg set size, singleton rate.
Hypothesis: weighted conformal gives equal coverage with smaller sets.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from smallmlp import SmallMLPClassifier
from smallmlp.conformal import conformal_qhat_classification, _to_tensor

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Datasets (same list as classification benchmark)
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
        for n_sub in [100, 200]:
            if len(y) >= n_sub:
                idx = rng.choice(len(y), size=n_sub, replace=False)
                datasets.append((f"{name}_{n_sub}", X[idx], y[idx]))

    return datasets


# ---------------------------------------------------------------------------
# Conformal helpers
# ---------------------------------------------------------------------------

def split_data(X, y, seed=42):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_cal, X_val, y_cal, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
    )
    return X_tr, y_tr, X_cal, y_cal, X_val, y_val


def conformal_global_q(scores_cal, alpha):
    n = len(scores_cal)
    target_q = float(np.ceil((1 - alpha) * (n + 1))) / n
    target_q = min(target_q, 1.0)
    return float(np.quantile(scores_cal, target_q))


def make_set(p, q_hat, fallback=True):
    """Prediction set from probability p and threshold q_hat."""
    s_0 = p
    s_1 = 1.0 - p
    c = set()
    if s_0 <= q_hat:
        c.add(0)
    if s_1 <= q_hat:
        c.add(1)
    if fallback and len(c) == 0:
        c.add(0 if s_0 < s_1 else 1)
    return c


def evaluate_sets(sets, y_val, classes):
    """Return coverage, avg size, singleton rate, empty rate."""
    # map y_val to binary
    y_bin = (y_val == classes[1]).astype(int)
    covered = np.array([y_bin[i] in sets[i] for i in range(len(y_val))])
    sizes = np.array([len(s) for s in sets])
    return (
        float(covered.mean()),
        float(sizes.mean()),
        float((sizes == 1).mean()),
        float((sizes == 0).mean()),
    )


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def run_smallmlp_wc(X_tr, y_tr, X_cal, y_cal, X_val, y_val, alpha):
    clf = SmallMLPClassifier(
        h_min=0.01, h_max=10.0,
        max_iter=30, inner_iter=10, tol=1e-8,
    )
    clf.fit(X_tr, y_tr)
    clf.fit_conformal(X_cal, y_cal, X_val, y_val, alpha=alpha)
    sets = clf.predict_set(X_val, alpha=alpha)
    return evaluate_sets(sets, y_val, clf.classes_)


def run_split_conformal_smallmlp(X_tr, y_tr, X_cal, y_cal, X_val, y_val, alpha):
    clf = SmallMLPClassifier(
        h_min=0.01, h_max=10.0,
        max_iter=30, inner_iter=10, tol=1e-8,
    )
    clf.fit(X_tr, y_tr)

    y_cal_bin = (y_cal == clf.classes_[1]).astype(np.float64)
    p_cal = clf._predict_p(X_cal)
    scores_cal = np.where(y_cal_bin == 1, 1.0 - p_cal, p_cal)
    q_g = conformal_global_q(scores_cal, alpha)

    p_val = clf._predict_p(X_val)
    sets = [make_set(p_val[i], q_g) for i in range(len(X_val))]
    return evaluate_sets(sets, y_val, clf.classes_)


def run_split_conformal_mlp(X_tr, y_tr, X_cal, y_cal, X_val, y_val, alpha):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=1000, random_state=42
        )),
    ])
    pipe.fit(X_tr, y_tr)
    classes = pipe.classes_

    p_cal = pipe.predict_proba(X_cal)[:, 1]
    y_cal_bin = (y_cal == classes[1]).astype(int)
    scores_cal = np.where(y_cal_bin == 1, 1.0 - p_cal, p_cal)
    q_g = conformal_global_q(scores_cal, alpha)

    p_val = pipe.predict_proba(X_val)[:, 1]
    sets = [make_set(p_val[i], q_g) for i in range(len(X_val))]
    return evaluate_sets(sets, y_val, classes)


def run_split_conformal_knn(X_tr, y_tr, X_cal, y_cal, X_val, y_val, alpha):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5)),
    ])
    pipe.fit(X_tr, y_tr)
    classes = pipe.classes_

    p_cal = pipe.predict_proba(X_cal)[:, 1]
    y_cal_bin = (y_cal == classes[1]).astype(int)
    scores_cal = np.where(y_cal_bin == 1, 1.0 - p_cal, p_cal)
    q_g = conformal_global_q(scores_cal, alpha)

    p_val = pipe.predict_proba(X_val)[:, 1]
    sets = [make_set(p_val[i], q_g) for i in range(len(X_val))]
    return evaluate_sets(sets, y_val, classes)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(alpha=0.1):
    datasets = build_datasets()

    print(f"\nDatasets: {len(datasets)}")
    print(f"alpha = {alpha}  (target coverage >= {1 - alpha:.2f})")
    print("=" * 110)

    methods = {
        "WC": run_smallmlp_wc,
        "SC_smallmlp": run_split_conformal_smallmlp,
        "SC_mlp": run_split_conformal_mlp,
        "SC_knn": run_split_conformal_knn,
    }

    rows = []
    for ds_name, X, y in datasets:
        X_tr, y_tr, X_cal, y_cal, X_val, y_val = split_data(X, y)
        row = {"dataset": ds_name, "n": len(y), "d": X.shape[1]}
        line = f"{ds_name:<32}"
        for mname, fn in methods.items():
            try:
                cov, size, sing, emp = fn(
                    X_tr, y_tr, X_cal, y_cal, X_val, y_val, alpha
                )
                row[f"{mname}_cov"] = cov
                row[f"{mname}_size"] = size
                row[f"{mname}_sing"] = sing
                row[f"{mname}_empty"] = emp
                line += f" | {mname}: cov={cov:.3f} sz={size:.2f}"
            except Exception as e:
                row[f"{mname}_cov"] = np.nan
                row[f"{mname}_size"] = np.nan
                row[f"{mname}_sing"] = np.nan
                row[f"{mname}_empty"] = np.nan
                line += f" | {mname}: FAIL"
        print(line)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize(df, alpha=0.1):
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)

    target = 1.0 - alpha
    methods = ["WC", "SC_smallmlp", "SC_mlp", "SC_knn"]

    print(f"\n{'method':<14} {'cov':>8} {'size':>8} {'sing':>8} {'empty':>8} "
          f"{'valid':>8} {'size_valid':>12}")
    print("-" * 80)
    for m in methods:
        cov = df[f"{m}_cov"].mean()
        size = df[f"{m}_size"].mean()
        sing = df[f"{m}_sing"].mean()
        emp = df[f"{m}_empty"].mean()
        valid = df[df[f"{m}_cov"] >= target - 0.02]
        n_valid = len(valid)
        size_valid = valid[f"{m}_size"].mean() if n_valid > 0 else np.nan
        print(f"{m:<14} {cov:>8.4f} {size:>8.3f} {sing:>8.3f} {emp:>8.3f} "
              f"{n_valid:>4}/{len(df):<3} {size_valid:>12.3f}")

    # head-to-head WC vs best split conformal
    both_valid = df[
        (df["WC_cov"] >= target - 0.02) &
        (df["SC_smallmlp_cov"] >= target - 0.02)
    ]
    if len(both_valid) > 0:
        print(f"\nHead-to-head WC vs SC_smallmlp (both valid, n={len(both_valid)}):")
        win_wc = (both_valid["WC_size"] < both_valid["SC_smallmlp_size"]).sum()
        win_sc = (both_valid["SC_smallmlp_size"] < both_valid["WC_size"]).sum()
        print(f"  WC smaller set: {win_wc}/{len(both_valid)}")
        print(f"  SC smaller set: {win_sc}/{len(both_valid)}")
        print(f"  mean size WC: {both_valid['WC_size'].mean():.3f}")
        print(f"  mean size SC: {both_valid['SC_smallmlp_size'].mean():.3f}")
        ratio = (both_valid["WC_size"] / both_valid["SC_smallmlp_size"]).mean()
        print(f"  ratio WC/SC: {ratio:.3f}  (<1 means WC better)")


if __name__ == "__main__":
    df = run_benchmark(alpha=0.1)
    df.to_csv("benchmarks/results_conformal_classification.csv", index=False)
    summarize(df, alpha=0.1)
    print("\nSaved: benchmarks/results_conformal_classification.csv")