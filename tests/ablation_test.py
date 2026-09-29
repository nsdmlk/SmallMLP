"""
Ablation: w_max formula for SmallMLPClassifier.

Modes:
  current   — min(4n, max(256, 32K))
  new       — min(8 * 0.8n / d, 3 * n/K, 4n)
  new_no4n  — min(8 * 0.8n / d, 3 * n/K)

Run:
  python ablation_wmax.py
"""

import time
import warnings

import numpy as np
from benchmarks.run_full_benchmark import build_classification_datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from smallmlp import SmallMLPClassifier
from smallmlp.nn import backbone as bb


# ----------------------------------------------------------------------
# w_max formulas
# ----------------------------------------------------------------------

def wmax_current(n, d, K):
    return min(4 * n, max(256, 32 * K))


def wmax_new(n, d, K):
    return min(8 * 0.8 * n / d, 3 * n / K, 4 * n)


def wmax_new_no4n(n, d, K):
    return min(8 * 0.8 * n / d, 3 * n / K)


# ----------------------------------------------------------------------
# Patch adaptive_width to use chosen w_max
# ----------------------------------------------------------------------

_ORIG_ADAPTIVE_WIDTH = bb.adaptive_width
_ACTIVE_WMAX_FN = wmax_current


def _patched_adaptive_width(n, d, K=2, l=1):
    w_max = _ACTIVE_WMAX_FN(n, d, K)
    if K <= 2:
        raw = np.sqrt(K) * np.log2(max(n, 2)) * (d / max(l, 1))
        return int(max(K, min(np.floor(raw), w_max)))
    K_lower = 2 * K
    n_per_class = max(n / K, 1.0)
    raw = (
        np.sqrt(K)
        * np.log2(n_per_class + 1.0)
        * (np.sqrt(d) / np.sqrt(max(l, 1)))
    )
    return int(max(K_lower, min(np.floor(raw), w_max)))


bb.adaptive_width = _patched_adaptive_width


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

def run(mode, datasets, n_splits=5, seed=42):
    global _ACTIVE_WMAX_FN
    if mode == "current":
        _ACTIVE_WMAX_FN = wmax_current
    elif mode == "new":
        _ACTIVE_WMAX_FN = wmax_new
    elif mode == "new_no4n":
        _ACTIVE_WMAX_FN = wmax_new_no4n
    else:
        raise ValueError(mode)

    accs = {}
    widths_seen = {}
    for name, X, y in datasets:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_accs = []
        for tr, te in skf.split(X, y):
            clf = SmallMLPClassifier(class_weight="balanced",
                                     max_epochs=500, patience=30,
                                     random_state=seed)
            try:
                clf.fit(X[tr], y[tr])
                acc = accuracy_score(y[te], clf.predict(X[te]))
                if name not in widths_seen:
                    widths_seen[name] = clf._model.widths
            except Exception as e:
                warnings.warn(f"{name} fold failed: {e}")
                acc = np.nan
            fold_accs.append(acc)
        accs[name] = float(np.nanmean(fold_accs))
    return accs, widths_seen


def _bucket(K):
    if K == 2:
        return "binary"
    if K <= 5:
        return "multiclass_2_5"
    return "multiclass_gt5"


def main():
    datasets = build_classification_datasets()
    print(f"Loaded {len(datasets)} datasets")

    # ---- sanity check: what w_max values are chosen ----
    print()
    print("w_max values per dataset:")
    print(f"{'dataset':<28} {'n':>5} {'d':>4} {'K':>3} "
          f"{'current':>10} {'new':>10} {'new_no4n':>10}")
    for name, X, y in datasets:
        n, d = X.shape[0], X.shape[1]
        K = len(np.unique(y))
        print(f"{name:<28} {n:>5} {d:>4} {K:>3} "
              f"{wmax_current(n, d, K):>10.1f} "
              f"{wmax_new(n, d, K):>10.1f} "
              f"{wmax_new_no4n(n, d, K):>10.1f}")

    # ---- run ----
    results = {}
    all_widths = {}
    for mode in ["current", "new", "new_no4n"]:
        t0 = time.time()
        accs, widths = run(mode, datasets, n_splits=5, seed=42)
        results[mode] = accs
        all_widths[mode] = widths
        print(f"[{mode}] done in {time.time()-t0:.1f}s", flush=True)

    Kmap = {name: len(np.unique(y)) for name, _, y in datasets}
    bmap = {name: _bucket(Kmap[name]) for name, _, _ in datasets}
    names = [d[0] for d in datasets]

    print()
    header = f"{'dataset':<28} {'K':>3} " + \
             " ".join(f"{m:>10}" for m in results)
    print(header)
    for name in names:
        row = f"{name:<28} {Kmap[name]:>3}"
        for mode in results:
            row += f" {results[mode][name]:>10.4f}"
        print(row)

    print()
    print("Overall mean:")
    for mode in results:
        vals = list(results[mode].values())
        print(f"  {mode:<10} {np.mean(vals):.4f}  (±{np.std(vals):.4f})")

    print()
    print("Per-bucket mean:")
    for b in ["binary", "multiclass_2_5", "multiclass_gt5"]:
        subset = [n for n in names if bmap[n] == b]
        if not subset:
            continue
        print(f"  [{b}]  n={len(subset)}")
        for mode in results:
            vals = [results[mode][n] for n in subset]
            print(f"    {mode:<10} {np.mean(vals):.4f}")

    print()
    print("Pairwise wins (row > col):")
    print(f"{'':>10} " + " ".join(f"{c:>10}" for c in results))
    for a in results:
        row = f"{a:>10} "
        for b in results:
            if a == b:
                row += f"{'—':>10} "
            else:
                wins = sum(1 for n in names if results[a][n] > results[b][n] + 1e-9)
                row += f"{wins:>10} "
        print(row)

    print()
    print("Widths sanity check (chosen widths, first fold):")
    for name in names:
        for mode in results:
            w = all_widths[mode].get(name)
            if w is not None:
                print(f"  {name:<28} {mode:<10} widths={w}")


if __name__ == "__main__":
    main()