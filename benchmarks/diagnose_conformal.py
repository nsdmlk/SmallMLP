"""Diagnose invalid conformal datasets and produce calibration plots.

- Lists datasets where WC coverage < target - 0.02, with n, d, n_cal.
- Calibration plot: empirical coverage vs nominal level for WC and SC.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from smallmlp import SmallMLPRegressor
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


def conformal_global_q(residuals_cal, alpha):
    n = len(residuals_cal)
    target_q = float(np.ceil((1 - alpha) * (n + 1))) / n
    target_q = min(target_q, 1.0)
    return float(np.quantile(residuals_cal, target_q))


# =====================================================================
# (A) Diagnose invalid datasets
# =====================================================================

def diagnose_invalid(alpha=0.1, coverage_slack=0.02):
    datasets = build_datasets()
    target = 1.0 - alpha

    print("=" * 100)
    print(f"INVALID DATASETS: WC coverage < {target - coverage_slack:.2f}")
    print("=" * 100)
    print(f"{'dataset':<28} {'n':>5} {'d':>4} {'n_tr':>5} {'n_cal':>6} "
          f"{'n_val':>6} {'cov_wc':>8} {'cov_sc':>8}")
    print("-" * 100)

    rows = []
    for ds_name, X, y in datasets:
        X_tr, y_tr, X_cal, y_cal, X_val, y_val = split_data(X, y)
        try:
            m = SmallMLPRegressor(
                h_min=0.01, h_max=10.0,
                max_iter=30, inner_iter=10, tol=1e-8,
            )
            m.fit(X_tr, y_tr)
            m.fit_conformal(X_cal, y_cal, X_val, y_val,
                            alpha=alpha, verbose=False)

            lo_wc, hi_wc = m.predict_interval_conformal(X_val, alpha=alpha)
            cov_wc = float(np.mean((y_val >= lo_wc) & (y_val <= hi_wc)))

            y_cal_hat = m.predict(X_cal)
            res_cal = np.abs(y_cal - y_cal_hat)
            q_g = conformal_global_q(res_cal, alpha)
            y_val_hat = m.predict(X_val)
            lo_sc, hi_sc = y_val_hat - q_g, y_val_hat + q_g
            cov_sc = float(np.mean((y_val >= lo_sc) & (y_val <= hi_sc)))

            rows.append({
                "dataset": ds_name,
                "n": len(y), "d": X.shape[1],
                "n_tr": len(y_tr), "n_cal": len(y_cal), "n_val": len(y_val),
                "cov_wc": cov_wc, "cov_sc": cov_sc,
            })
        except Exception as e:
            print(f"{ds_name:<28} FAILED: {e}")

    df = pd.DataFrame(rows)
    invalid = df[df["cov_wc"] < target - coverage_slack]
    valid = df[df["cov_wc"] >= target - coverage_slack]

    for _, r in df.iterrows():
        marker = " <<< INVALID" if r["cov_wc"] < target - coverage_slack else ""
        print(f"{r['dataset']:<28} {int(r['n']):>5} {int(r['d']):>4} "
              f"{int(r['n_tr']):>5} {int(r['n_cal']):>6} {int(r['n_val']):>6} "
              f"{r['cov_wc']:>8.3f} {r['cov_sc']:>8.3f}{marker}")

    print("\n" + "-" * 100)
    print(f"INVALID:  {len(invalid)} / {len(df)}")
    print(f"VALID:    {len(valid)} / {len(df)}")

    if len(invalid) > 0:
        print("\nInvalid dataset stats:")
        print(f"  mean n:     {invalid['n'].mean():.1f}")
        print(f"  mean n_cal: {invalid['n_cal'].mean():.1f}")
        print(f"  mean d:     {invalid['d'].mean():.1f}")

    if len(valid) > 0:
        print("\nValid dataset stats:")
        print(f"  mean n:     {valid['n'].mean():.1f}")
        print(f"  mean n_cal: {valid['n_cal'].mean():.1f}")
        print(f"  mean d:     {valid['d'].mean():.1f}")

    # correlation: n_cal vs coverage
    if len(df) > 1:
        corr = df["n_cal"].corr(df["cov_wc"])
        print(f"\nCorrelation(n_cal, coverage_wc): {corr:+.3f}")

    df.to_csv("benchmarks/diagnose_invalid.csv", index=False)
    return df


# =====================================================================
# (B) Calibration plot: coverage vs nominal
# =====================================================================

def calibration_plot(levels=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    datasets = build_datasets()

    print("\n" + "=" * 100)
    print("CALIBRATION: empirical coverage vs nominal level")
    print("=" * 100)

    records = []
    for ds_name, X, y in datasets:
        X_tr, y_tr, X_cal, y_cal, X_val, y_val = split_data(X, y)
        try:
            m = SmallMLPRegressor(
                h_min=0.01, h_max=10.0,
                max_iter=30, inner_iter=10, tol=1e-8,
            )
            m.fit(X_tr, y_tr)

            for level in levels:
                alpha = 1.0 - level
                try:
                    m.fit_conformal(X_cal, y_cal, X_val, y_val,
                                    alpha=alpha, verbose=False)
                    lo_wc, hi_wc = m.predict_interval_conformal(X_val, alpha=alpha)
                    cov_wc = float(np.mean((y_val >= lo_wc) & (y_val <= hi_wc)))

                    y_cal_hat = m.predict(X_cal)
                    res_cal = np.abs(y_cal - y_cal_hat)
                    q_g = conformal_global_q(res_cal, alpha)
                    y_val_hat = m.predict(X_val)
                    cov_sc = float(np.mean(
                        (y_val >= y_val_hat - q_g) & (y_val <= y_val_hat + q_g)
                    ))

                    records.append({
                        "dataset": ds_name,
                        "level": level,
                        "cov_wc": cov_wc,
                        "cov_sc": cov_sc,
                    })
                except Exception:
                    pass
        except Exception:
            pass

    df = pd.DataFrame(records)
    df.to_csv("benchmarks/calibration_data.csv", index=False)

    # aggregate
    agg = df.groupby("level")[["cov_wc", "cov_sc"]].mean()
    print("\nMean empirical coverage:")
    print(f"{'level':>6} {'WC':>8} {'SC':>8}")
    for level, row in agg.iterrows():
        print(f"{level:>6.2f} {row['cov_wc']:>8.3f} {row['cov_sc']:>8.3f}")

    # plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="ideal", linewidth=1)
    ax.plot(agg.index, agg["cov_wc"], "o-", label="Weighted conformal", linewidth=2)
    ax.plot(agg.index, agg["cov_sc"], "s-", label="Split conformal", linewidth=2)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage (mean over datasets)")
    ax.set_title("Calibration plot")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    plt.tight_layout()
    plt.savefig("benchmarks/calibration_plot.png", dpi=150)
    print("\nSaved: benchmarks/calibration_plot.png")
    print("Saved: benchmarks/calibration_data.csv")


if __name__ == "__main__":
    diagnose_invalid()
    calibration_plot()