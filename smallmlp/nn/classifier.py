import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .backbone import _Backbone, adaptive_dropout
from ..conformal import conformal_qhat, _get_embedding
from ..conformal.weighted import tune_h_cal_classification


class SmallMLPClassifier(BaseEstimator, ClassifierMixin):
    """Adaptive MLP for small classification (binary or multiclass) with
    weighted conformal prediction sets.

    width_mode:
      'formula'  — w_1 = max(2K, min(floor(alpha * sqrt(n * K * sqrt(d))), 4n))
                   w_l = max(K, floor(w_1 * beta^(l-1)))
                   alpha=4.0, beta=0.7 by default.
      'classic'  — min(2n, 128) for all layers

    class_weight:
      None       — no class weighting (default, best for accuracy)
      'balanced' — sqrt-balanced weights, applied only when ratio > 2.
                   Improves rare-class recall at the cost of accuracy.

    bias_init:
      'kaiming'  — PyTorch default (uniform, centered ~0)
      'zeros'    — bias = 0
      'positive' — bias = +0.1 (keeps ReLU in active region)
      'uniform'  — bias ~ U(0, 0.5)
    """

    def __init__(self, activation="relu", lr=1e-3, weight_decay=None,
                 max_epochs=500, patience=30, batch_size=None,
                 val_frac=0.2, class_weight=None, random_state=42,
                 verbose=False, width_mode="formula",
                 alpha=4.0, beta=0.7, bias_init="kaiming"):
        self.activation = activation
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose
        self.width_mode = width_mode
        self.alpha = alpha
        self.beta = beta
        self.bias_init = bias_init

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError(
                f"SmallMLPClassifier requires at least 2 classes, "
                f"got {self.n_classes_}."
            )

        self._label_encoder = LabelEncoder().fit(self.classes_)
        y_enc = self._label_encoder.transform(y).astype(np.int64)
        n = X.shape[0]

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._x_scaler = StandardScaler().fit(X)
        Xs = self._x_scaler.transform(X)

        dropout = adaptive_dropout(n, self.n_features_in_)
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

        self._model = _Backbone(
            self.n_features_in_, n, K=self.n_classes_,
            dropout=dropout, activation=self.activation,
            width_mode=self.width_mode,
            alpha=self.alpha, beta=self.beta,
            bias_init=self.bias_init,
        )
        out_dim = self._model.output_dim

        if self.verbose:
            print(f"[SmallMLPClassifier] n={n} d={self.n_features_in_} "
                  f"K={self.n_classes_} width_mode={self.width_mode} "
                  f"alpha={self.alpha} beta={self.beta} "
                  f"bias_init={self.bias_init} "
                  f"dropout={dropout:.3f} wd={wd:.4f}")
            print(f"  widths={self._model.widths}")

        self._head = nn.Linear(out_dim, self.n_classes_)
        params = list(self._model.parameters()) + list(self._head.parameters())

        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=wd)

        if self.class_weight == "balanced":
            counts = np.bincount(y_enc, minlength=self.n_classes_).astype(np.float64)
            counts = np.maximum(counts, 1.0)
            ratio = counts.max() / counts.min()
            if ratio > 2.0:
                # sqrt-balanced: tames extreme imbalance (ratio > 10),
                # where linear weights (n / K*n_k) destabilize training.
                weights = np.sqrt(n / (self.n_classes_ * counts))
                weights = weights / weights.mean()  # normalize to mean 1
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
            self._model.train()
            self._head.train()
            perm = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), bs):
                b = perm[i:i + bs]
                optimizer.zero_grad()
                h = self._model(X_tr[b])
                logits = self._head(h)
                loss = loss_fn(logits, y_tr[b])
                loss.backward()
                optimizer.step()

            self._model.eval()
            self._head.eval()
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
                    if self.verbose:
                        print(f"  early stop at epoch {epoch}")
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state["model"])
            self._head.load_state_dict(best_state["head"])

        self._model.eval()
        self._head.eval()
        self._loss_ = best_val
        return self

    def _prepare_query(self, X):
        check_is_fitted(self, ["_model", "_head"])
        X = check_array(X, dtype=np.float64)
        Xs = self._x_scaler.transform(X)
        return torch.tensor(Xs, dtype=torch.float32)

    def predict_proba(self, X):
        X_t = self._prepare_query(X)
        with torch.no_grad():
            h = self._model(X_t)
            logits = self._head(h)
            p = torch.softmax(logits, dim=-1).numpy()
        p = np.clip(p, 1e-12, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def predict(self, X):
        p = self.predict_proba(X)
        idx = np.argmax(p, axis=1)
        return self.classes_[idx]

    # ---------------- weighted conformal ----------------

    def fit_conformal(self, X_cal, y_cal, X_val, y_val,
                      alpha=0.1, h_grid=None, verbose=False):
        check_is_fitted(self, ["_model", "_head"])

        best = tune_h_cal_classification(
            self, X_cal, y_cal, X_val, y_val,
            alpha=alpha, h_grid=h_grid,
        )

        X_cal_emb, _ = _get_embedding(self, X_cal)
        self._h_cal = np.full(X_cal_emb.shape[1], best["h_cal"])
        self._alpha_conformal = alpha
        self._X_cal_emb = X_cal_emb

        y_cal_enc = self._label_encoder.transform(y_cal).astype(np.int64)
        p_cal = self.predict_proba(X_cal)
        self._scores_cal = 1.0 - p_cal[np.arange(len(y_cal_enc)), y_cal_enc]

        if verbose:
            print(f"[conformal] h_cal={best['h_cal']:.4f}  "
                  f"val_cov={best['coverage']:.4f}  "
                  f"val_size={best['size']:.4f}")
        return self

    def predict_set(self, X, alpha=None):
        check_is_fitted(self, ["_scores_cal"])
        if alpha is None:
            alpha = self._alpha_conformal

        X_emb, _ = _get_embedding(self, X)
        X_emb_t = torch.tensor(X_emb, dtype=torch.float32)
        X_cal_emb_t = torch.tensor(self._X_cal_emb, dtype=torch.float32)
        scores_t = torch.tensor(self._scores_cal, dtype=torch.float32)
        h_cal_t = torch.tensor(self._h_cal, dtype=torch.float32)

        q_hat = conformal_qhat(
            X_emb_t, X_cal_emb_t, scores_t, h_cal_t, alpha
        )

        probs = self.predict_proba(X)
        sets = []
        for i in range(len(X)):
            c = set()
            for k in range(self.n_classes_):
                if 1.0 - probs[i, k] <= q_hat[i]:
                    c.add(k)
            if len(c) == 0:
                c.add(int(np.argmax(probs[i])))
            sets.append(c)
        return sets

    def predict_set_labels(self, X, alpha=None):
        sets = self.predict_set(X, alpha=alpha)
        return [{self.classes_[k] for k in s} for s in sets]