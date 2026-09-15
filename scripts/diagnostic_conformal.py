"""Diagnostic script for weighted conformal. Not part of the library.

Run: python scripts/debug_conformal.py
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from smallmlp import SmallMLPRegressor
from smallmlp.conformal import conformal_qhat, _to_tensor


def percentile_row(name, arr):
    p = np.percentile(arr, [10, 25, 50, 75, 90, 100])
    return f"{name:<28} " + "  ".join(f"{v:>9.2f}" for v in p)


def main():
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=0
    )
    X_cal, X_val, y_cal, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=0
    )

    m = SmallMLPRegressor()
    m.fit(X_tr, y_tr)
    m.fit_conformal(X_cal, y_cal, X_val, y_val, alpha=0.1, verbose=True)

    # --- residuals ---
    y_cal_hat = m.predict(X_cal)
    y_val_hat = m.predict(X_val)
    res_cal = np.abs(y_cal - y_cal_hat)
    res_val = np.abs(y_val - y_val_hat)

    print("\n--- residuals ---")
    print(" " * 28 + "  ".join(f"{p:>9}" for p in ["p10", "p25", "p50", "p75", "p90", "max"]))
    print(percentile_row("cal residuals", res_cal))
    print(percentile_row("val residuals", res_val))

    # --- q_hat ---
    X_val_t = _to_tensor(m, X_val)
    X_cal_t = _to_tensor(m, m._X_cal_raw)
    residuals_t = __import__("torch").tensor(m._residuals_cal, dtype=__import__("torch").float64)
    h_cal_t = __import__("torch").tensor(m._h_cal, dtype=__import__("torch").float64)

    q_hat = conformal_qhat(X_val_t, X_cal_t, residuals_t, h_cal_t, 0.1)

    print("\n--- q_hat ---")
    print(percentile_row("q_hat", q_hat))

    # --- interval ---
    lo, hi = m.predict_interval_conformal(X_val, alpha=0.1)
    widths = hi - lo
    print("\n--- interval ---")
    print(percentile_row("width", widths))

    coverage = float(np.mean((y_val >= lo) & (y_val <= hi)))
    print(f"\ncoverage = {coverage:.4f}  (target >= 0.90)")

    # --- per-point diagnostics (first 10) ---
    print("\n--- per-point (first 10) ---")
    print(f"{'i':>3}  {'y_true':>8}  {'y_hat':>8}  {'res_val':>8}  {'q_hat':>8}  {'in?':>4}")
    for i in range(min(10, len(y_val))):
        in_interval = lo[i] <= y_val[i] <= hi[i]
        print(f"{i:>3}  {y_val[i]:>8.2f}  {y_val_hat[i]:>8.2f}  "
              f"{res_val[i]:>8.2f}  {q_hat[i]:>8.2f}  {str(in_interval):>4}")


if __name__ == "__main__":
    main()