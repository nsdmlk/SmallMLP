"""
Ablation: current dropout formula vs clamp formula vs fixed 0.2.

Modes:
  current   — min(d/n, 0.5)                          (baseline)
  clamp     — clamp(3*d/n, 0.1, 0.5)                 (new)
  fixed_0.2 — 0.2                                    (reference upper bound)

Run:
  python ablation_dropout_v2.py
"""

import time
import warnings

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from smallmlp import SmallMLPClassifier
from smallmlp.nn import backbone as bb


# ----------------------------------------------------------------------
# Candidate dropout formulas
# ----------------------------------------------------------------------

def current_dropout(n, d, p_max=0.5):
    return float(min(max(d / max(n, 1), 0.0), p_max))


def clamp_dropout(n, d, c=3.0, p_min=0.1, p_max=0.5):
    return float(min(max(c * d / max(n, 1), p_min), p_max))


# ----------------------------------------------------------------------
# Patch _Backbone to accept dropout_override
# ----------------------------------------------------------------------

_ORIG_INIT = bb._Backbone.__init__


def _patched_init(self, d_in, n, K=2, dropout=0.0, activation="relu",
                  n_layers=2, width_mode="formula", dropout_override=None):
    if dropout_override is not None:
        dropout = dropout_override
    _ORIG_INIT(self, d_in, n, K=K, dropout=dropout, activation=activation,
               n_layers=n_layers, width_mode=width_mode)


bb._Backbone.__init__ = _patched_init

import smallmlp.nn.classifier as cls_mod


# ----------------------------------------------------------------------
# Patched fit — same as previous ablation, passes dropout_override
# ----------------------------------------------------------------------

def _patched_fit(self, X, y):
    from sklearn.utils.validation import check_X_y
    from sklearn.utils.multiclass import check_classification_targets
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import torch
    import torch.nn as nn

    X, y = check_X_y(X, y)
    check_classification_targets(y)
    self.n_features_in_ = X.shape[1]
    self.classes_ = np.unique(y)
    self.n_classes_ = len(self.classes_)
    if self.n_classes_ < 2:
        raise ValueError("Need >= 2 classes")

    self._label_encoder = LabelEncoder().fit(self.classes_)
    y_enc = self._label_encoder.transform(y).astype(np.int64)
    n = X.shape[0]

    torch.manual_seed(self.random_state)
    np.random.seed(self.random_state)

    self._x_scaler = StandardScaler().fit(X)
    Xs = self._x_scaler.transform(X)

    dropout = bb.adaptive_dropout(n, self.n_features_in_)
    wd = self.weight_decay if self.weight_decay is not None else 1.0 / n

    rng = np.random.default_rng(self.random_state)
    idx = rng.permutation(n)
    n_val = max(int(self.val_frac * n), 1)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    X_tr = torch.tensor(Xs[tr_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_enc[tr_idx], dtype=torch.long)
    X_val = torch.tensor(Xs[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y_enc[val_idx], dtype=torch.long)

    self._model = bb._Backbone(
        self.n_features_in_, n, K=self.n_classes_,
        dropout=dropout, activation=self.activation,
        width_mode=self.width_mode,
        dropout_override=getattr(self, "_dropout_override", None),
    )
    out_dim = self._model.output_dim

    self._head = nn.Linear(out_dim, self.n_classes_)
    params = list(self._model.parameters()) + list(self._head.parameters())
    optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=wd)

    if self.class_weight == "balanced":
        counts = np.bincount(y_enc, minlength=self.n_classes_).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        ratio = counts.max() / counts.min()
        if ratio > 2.0:
            weights = n / (self.n_classes_ * counts)
            class_weights = torch.tensor(weights, dtype=torch.float32)
        else:
            class_weights = None
    else:
        class_weights = None

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    bs = self.batch_size or min(n, 32)
    best_val = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(self.max_epochs):
        self._model.train(); self._head.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), bs):
            b = perm[i:i + bs]
            optimizer.zero_grad()
            h = self._model(X_tr[b])
            logits = self._head(h)
            loss = loss_fn(logits, y_tr[b])
            loss.backward()
            optimizer.step()

        self._model.eval(); self._head.eval()
        with torch.no_grad():
            h_val = self._model(X_val)
            val_logits = self._head(h_val)
            val_loss = loss_fn(val_logits, y_val).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {
                "model": {k: v.clone() for k, v in self._model.state_dict().items()},
                "head": {k: v.clone() for k, v in self._head.state_dict().items()},
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= self.patience:
                break

    if best_state is not None:
        self._model.load_state_dict(best_state["model"])
        self._head.load_state_dict(best_state["head"])
    self._model.eval(); self._head.eval()
    self._loss_ = best_val
    return self


cls_mod.SmallMLPClassifier.fit = _patched_fit


# ----------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------

def _subsample(X, y, n_max, seed=0):
    if X.shape[0] <= n_max:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], n_max, replace=False)
    return X[idx], y[idx]


def load_datasets(n_max=500):
    out = []
    for name, loader in [
        ("iris", load_iris),
        ("wine", load_wine),
        ("breast_cancer", load_breast_cancer),
    ]:
        d = loader()
        X, y = _subsample(d.data.astype(np.float64), d.target.astype(np.int64), n_max)
        out.append((name, X, y))

    for id_, name in [
        (53,   "vehicle"),
        (41,   "glass"),
        (181,  "yeast"),
        (23,   "cmc"),
        (36,   "segment"),
        (1480, "vowel"),
        (1056, "mfeat_fourier"),
    ]:
        try:
            ds = fetch_openml(data_id=id_, as_frame=False, parser="auto")
            X = np.asarray(ds.data, dtype=np.float64)
            y = LabelEncoder().fit_transform(ds.target).astype(np.int64)
            X, y = _subsample(X, y, n_max)
            out.append((name, X, y))
        except Exception as e:
            warnings.warn(f"skip {name}: {e}")
    return out


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

def run(mode, datasets, n_splits=10, seed=0):
    """mode: 'current' | 'clamp' | 'fixed_0.2' | 'fixed_0.1' | 'fixed_0.3'."""
    accs = {}
    ps = {}
    for name, X, y in datasets:
        n, d = X.shape[0], X.shape[1]
        if mode == "current":
            override = None  # use backbone's adaptive_dropout (current)
            p_val = current_dropout(n, d)
        elif mode == "clamp":
            p_val = clamp_dropout(n, d)
            override = p_val
        elif mode == "fixed_0.2":
            override = 0.2
            p_val = 0.2
        elif mode == "fixed_0.1":
            override = 0.1
            p_val = 0.1
        elif mode == "fixed_0.3":
            override = 0.3
            p_val = 0.3
        else:
            raise ValueError(mode)
        ps[name] = p_val

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_accs = []
        for tr, te in skf.split(X, y):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            X_tr2, X_val, y_tr2, y_val = train_test_split(
                X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=seed
            )
            clf = SmallMLPClassifier(random_state=seed, width_mode="formula")
            clf._dropout_override = override
            try:
                clf.fit(X_tr2, y_tr2)
                acc = (clf.predict(X_te) == y_te).mean()
            except Exception as e:
                warnings.warn(f"{name} fold failed: {e}")
                acc = np.nan
            fold_accs.append(acc)
        accs[name] = float(np.nanmean(fold_accs))
    return accs, ps


def main():
    datasets = load_datasets()
    print(f"Loaded {len(datasets)} datasets")

    modes = ["current", "clamp", "fixed_0.1", "fixed_0.2", "fixed_0.3"]

    results = {}
    all_ps = {}
    for mode in modes:
        t0 = time.time()
        accs, ps = run(mode, datasets, n_splits=10, seed=0)
        results[mode] = accs
        all_ps[mode] = ps
        print(f"[{mode}] done in {time.time()-t0:.1f}s", flush=True)

    Kmap = {name: len(np.unique(y)) for name, _, y in datasets}

    print()
    print("Dropout values per dataset:")
    header = f"{'dataset':<18} {'n':>4} {'d':>4} " + \
             " ".join(f"{m:>10}" for m in modes)
    print(header)
    for name, X, y in datasets:
        row = f"{name:<18} {X.shape[0]:>4} {X.shape[1]:>4}"
        for mode in modes:
            row += f" {all_ps[mode][name]:>10.4f}"
        print(row)

    print()
    header = f"{'dataset':<18} {'K':>3} " + " ".join(f"{m:>10}" for m in modes)
    print(header)
    for name, X, y in datasets:
        row = f"{name:<18} {Kmap[name]:>3}"
        for mode in modes:
            row += f" {results[mode][name]:>10.4f}"
        print(row)

    print()
    print("Overall mean:")
    for mode in modes:
        vals = list(results[mode].values())
        print(f"  {mode:<12} {np.mean(vals):.4f}  (±{np.std(vals):.4f})")

    print()
    print("Pairwise wins (row > col):")
    print(f"{'':>12} " + " ".join(f"{c:>10}" for c in modes))
    names = [d[0] for d in datasets]
    for a in modes:
        row = f"{a:>12} "
        for b in modes:
            if a == b:
                row += f"{'—':>10} "
            else:
                wins = sum(1 for n in names if results[a][n] > results[b][n] + 1e-9)
                row += f"{wins:>10} "
        print(row)

    print()
    print("Best mode per dataset:")
    for name in names:
        best = max(modes, key=lambda m: results[m][name])
        print(f"  {name:<18} best={best:<12} acc={results[best][name]:.4f}")


if __name__ == "__main__":
    main()