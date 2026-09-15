"""Benchmark: weighted conformal vs split conformal vs heuristic zone.

Splits each dataset 60/20/20 (train / calibration / validation).
For alpha = 0.1, reports coverage and mean width on validation.
Hypothesis: weighted conformal gives equal coverage with smaller width.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from smallmlp import SmallMLPRegressor
from smallmlp.conformal import conformal_qhat, _to_tensor
from benchmarks.run_mlp_benchmark import build_datasets

warnings.filterwarnings("ignore")


def split_data(X, y, seed=42):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=seed
    )
    X_cal, X_val, y_cal, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed
    )
    return X_tr, y_tr, X_cal, y_cal, X_val, y_val


def split_conformal_qhat(residuals_cal, alpha):
    """Global (unweighted) conformal quantile."""
    n = len(residuals_cal)
    target_q = float(np.ceil((1 - alpha) * (n + 1))) / n
    target_q = min(target_q, 1.0)
    return float(np.quantile(residuals_cal, target_q))


def evaluate_method(lo, hi, y_val):
    coverage = float(np.mean((y_val >= lo) & (y_val <= hi)))
    width = float(np.mean(hi - lo))
    return coverage, width


def run_benchmark(alpha=0.1):
    datasets = build_datasets()

    print(f"Datasets: {len(datasets)}")
    print(f"alpha = {alpha}  (target coverage >= {1 - alpha:.2f})")
    print("=" * 90)

    rows = []
    for ds_name, X, y in datasets:
        try:
            X_tr, y_tr, X_cal, y_cal, X_val, y_val = split_data(X, y)

            # --- fit point predictor ---
            m = SmallMLPRegressor(
                h_min=0.01, h_max=10.0,
                max_iter=30, inner_iter=10, tol=1e-8,
            )
            m.fit(X_tr, y_tr)

            # --- weighted conformal ---
            try:
                m.fit_conformal(
                    X_cal, y_cal, X_val, y_val,
                    alpha=alpha, verbose=False,
                )
                lo_wc, hi_wc = m.predict_interval_conformal(X_val, alpha=alpha)
                cov_wc, w_wc = evaluate_method(lo_wc, hi_wc, y_val)
            except Exception as e:
                cov_wc, w_wc = np.nan, np.nan

            # --- split conformal (global quantile) ---
            try:
                y_cal_hat = m.predict(X_cal)
                res_cal = np.abs(y_cal - y_cal_hat)
                q_global = split_conformal_qhat(res_cal, alpha)
                y_val_hat = m.predict(X_val)
                lo_sc = y_val_hat - q_global
                hi_sc = y_val_hat + q_global
                cov_sc, w_sc = evaluate_method(lo_sc, hi_sc, y_val)
            except Exception:
                cov_sc, w_sc = np.nan, np.nan

            # --- heuristic zone ---
            try:
                lo_hz, hi_hz = m.predict_interval(X_val, alpha=1 - alpha)
                cov_hz, w_hz = evaluate_method(lo_hz, hi_hz, y_val)
            except Exception:
                cov_hz, w_hz = np.nan, np.nan

            rows.append({
                "dataset": ds_name,
                "n": len(y),
                "d": X.shape[1],
                "cov_wc": cov_wc, "width_wc": w_wc,
                "cov_sc": cov_sc, "width_sc": w_sc,
                "cov_hz": cov_hz, "width_hz": w_hz,
            })

            print(f"{ds_name:28s}  "
                  f"WC cov={cov_wc:.3f} w={w_wc:8.3f} | "
                  f"SC cov={cov_sc:.3f} w={w_sc:8.3f} | "
                  f"HZ cov={cov_hz:.3f} w={w_hz:8.3f}")

        except Exception as e:
            print(f"{ds_name:28s}  FAILED: {e}")

    df = pd.DataFrame(rows)
    return df


def summarize(df):
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    methods = [("WC", "cov_wc", "width_wc"),
               ("SC", "cov_sc", "width_sc"),
               ("HZ", "cov_hz", "width_hz")]

    print("\nMean coverage / mean width (lower width is better at equal coverage):")
    for name, cov_col, w_col in methods:
        cov = df[cov_col].mean()
        w = df[width_col].mean() if False else df[w_col].mean()
        print(f"  {name:4s}:  coverage={cov:.4f}  width={w:8.3f}")

    # valid = coverage within 2% of target
    target = 1.0 - 0.1
    print(f"\nMean width among VALID methods (coverage >= {target - 0.02:.2f}):")
    for name, cov_col, w_col in methods:
        valid = df[df[cov_col] >= target - 0.02]
        if len(valid) > 0:
            print(f"  {name:4s}:  n_valid={len(valid):3d}  "
                  f"mean_width={valid[w_col].mean():8.3f}  "
                  f"mean_cov={valid[cov_col].mean():.4f}")
        else:
            print(f"  {name:4s}:  no valid datasets")

    # head-to-head: WC vs SC on width, among datasets where both valid
    both_valid = df[
        (df["cov_wc"] >= target - 0.02) & (df["cov_sc"] >= target - 0.02)
    ]
    if len(both_valid) > 0:
        win_wc = (both_valid["width_wc"] < both_valid["width_sc"]).sum()
        win_sc = (both_valid["width_sc"] < both_valid["width_wc"]).sum()
        print(f"\nHead-to-head WC vs SC (width, both valid):")
        print(f"  WC narrower: {win_wc}/{len(both_valid)}")
        print(f"  SC narrower: {win_sc}/{len(both_valid)}")
        print(f"  mean width WC: {both_valid['width_wc'].mean():.3f}")
        print(f"  mean width SC: {both_valid['width_sc'].mean():.3f}")
        rel = (both_valid["width_wc"] / both_valid["width_sc"]).mean()
        print(f"  ratio WC/SC: {rel:.3f}  (<1 means WC better)")

    return df


if __name__ == "__main__":
    df = run_benchmark(alpha=0.1)
    df.to_csv("benchmarks/results_conformal.csv", index=False)
    summarize(df)
    print("\nSaved: benchmarks/results_conformal.csv")