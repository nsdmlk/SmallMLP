"""
Ablation: adaptive alpha for conformal regression.

alpha_eff = alpha * max(1, sqrt(threshold / (n/d)))

Modes:
  baseline  — alpha_eff = alpha (no adaptation)
  th20      — threshold=20, sqrt
  th25      — threshold=25, sqrt
  th20_lin  — threshold=20, linear
"""

import warnings
import numpy as np
from sklearn.model_selection import train_test_split

from smallmlp import SmallMLPRegressor
from benchmarks.run_full_benchmark import build_regression_datasets

warnings.filterwarnings("ignore")


def effective_alpha(alpha, n, d, threshold, mode="sqrt"):
    ratio = n / max(d, 1)
    if ratio >= threshold:
        return alpha
    factor = threshold / ratio
    if mode == "sqrt":
        factor = np.sqrt(factor)
    return alpha * factor


def run_mode(mode, threshold, datasets, base_alpha=0.1):
    rows = []
    for name, X, y in datasets:
        try:
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                X, y, test_size=0.4, random_state=0
            )
            X_cal, X_val, y_cal, y_val = train_test_split(
                X_tmp, y_tmp, test_size=0.5, random_state=0
            )
            n, d = X.shape
            if mode == "baseline":
                a_eff = base_alpha
            else:
                a_eff = effective_alpha(base_alpha, n, d, threshold, mode)

            model = SmallMLPRegressor(max_epochs=500, patience=30)
            model.fit(X_tr, y_tr)
            model.fit_conformal(X_cal, y_cal, X_val, y_val, alpha=a_eff)

            lo, hi = model.predict_interval_conformal(X_val, alpha=a_eff)
            cov = float(np.mean((y_val >= lo) & (y_val <= hi)))
            width = float(np.mean(hi - lo))

            rows.append({
                "dataset": name, "n": n, "d": d, "n_over_d": n / d,
                "alpha_eff": a_eff, "coverage": cov, "width": width,
                "valid": int(cov >= 0.88),
            })
        except Exception as e:
            print(f"{name}: {e}")
    return rows


def summarize(rows, label):
    valid = sum(r["valid"] for r in rows)
    coverages = [r["coverage"] for r in rows]
    widths = [r["width"] for r in rows]
    valid_widths = [r["width"] for r in rows if r["valid"]]
    print(f"\n[{label}]")
    print(f"  Valid: {valid}/{len(rows)}")
    print(f"  Mean coverage: {np.mean(coverages):.3f}")
    print(f"  Mean width (all): {np.mean(widths):.3f}")
    print(f"  Mean width (valid only): {np.mean(valid_widths):.3f}")
    return valid, np.mean(valid_widths)


def main():
    datasets = build_regression_datasets()
    print(f"Loaded {len(datasets)} datasets")

    results = {}

    print("\n=== baseline ===")
    rows = run_mode("baseline", None, datasets)
    results["baseline"] = summarize(rows, "baseline")

    for threshold in [15, 20, 25, 30]:
        print(f"\n=== sqrt, threshold={threshold} ===")
        rows = run_mode("sqrt", threshold, datasets)
        results[f"sqrt_t{threshold}"] = summarize(rows, f"sqrt, t={threshold}")

    for threshold in [20, 25]:
        print(f"\n=== linear, threshold={threshold} ===")
        rows = run_mode("linear", threshold, datasets)
        results[f"lin_t{threshold}"] = summarize(rows, f"linear, t={threshold}")

    print("\n=== summary ===")
    print(f"{'mode':<16} {'valid':>6} {'mean_width_valid':>18}")
    for label, (v, w) in results.items():
        print(f"{label:<16} {v:>6} {w:>18.3f}")


if __name__ == "__main__":
    main()