import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler

from .backbone import _Backbone, adaptive_dropout
from ..conformal import conformal_qhat, _get_embedding
from ..conformal.weighted import tune_h_cal_regression


class SmallMLPRegressor(BaseEstimator, RegressorMixin):
    """Adaptive MLP for small regression with weighted conformal intervals."""

    def __init__(self, activation="relu", lr=1e-3, weight_decay=None,
                 max_epochs=500, patience=30, batch_size=None,
                 val_frac=0.2, random_state=42, verbose=False):
        self.activation = activation
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n = X.shape[0]

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._x_scaler = StandardScaler().fit(X)
        Xs = self._x_scaler.transform(X)
        self._y_mean = float(y.mean())
        y_std = float(y.std()) or 1.0
        self._y_std = y_std
        ys = (y - self._y_mean) / self._y_std

        dropout = adaptive_dropout(n, self.n_features_in_)
        wd = self.weight_decay if self.weight_decay is not None else 1.0 / n

        if self.verbose:
            print(f"[SmallMLPRegressor] n={n} d={self.n_features_in_} "
                  f"dropout={dropout:.3f} wd={wd:.4f}")

        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(n)
        n_val = max(int(self.val_frac * n), 1)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        X_tr = torch.tensor(Xs[tr_idx], dtype=torch.float32)
        y_tr = torch.tensor(ys[tr_idx], dtype=torch.float32)
        X_val = torch.tensor(Xs[val_idx], dtype=torch.float32)
        y_val = torch.tensor(ys[val_idx], dtype=torch.float32)

        self._model = _Backbone(
            self.n_features_in_, n, K=2,
            dropout=dropout, activation=self.activation,
            width_mode="classic",
        )
        out_dim = self._model.output_dim

        if self.verbose:
            print(f"  widths={self._model.widths}")

        self._head = nn.Linear(out_dim, 1)
        params = list(self._model.parameters()) + list(self._head.parameters())

        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=wd)
        loss_fn = nn.MSELoss()

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
                pred = self._head(h).squeeze(-1)
                loss = loss_fn(pred, y_tr[b])
                loss.backward()
                optimizer.step()

            self._model.eval()
            self._head.eval()
            with torch.no_grad():
                h_val = self._model(X_val)
                val_pred = self._head(h_val).squeeze(-1)
                val_loss = loss_fn(val_pred, y_val).item()

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

    def predict(self, X):
        X_t = self._prepare_query(X)
        with torch.no_grad():
            h = self._model(X_t)
            pred = self._head(h).squeeze(-1).numpy()
        return pred * self._y_std + self._y_mean

    def fit_conformal(self, X_cal, y_cal, X_val, y_val,
                      alpha=0.1, h_grid=None, verbose=False):
        check_is_fitted(self, ["_model", "_head"])

        best = tune_h_cal_regression(
            self, X_cal, y_cal, X_val, y_val,
            alpha=alpha, h_grid=h_grid,
        )

        X_cal_emb, _ = _get_embedding(self, X_cal)
        self._h_cal = np.full(X_cal_emb.shape[1], best["h_cal"])
        self._alpha_conformal = alpha
        self._X_cal_emb = X_cal_emb

        y_cal_hat = self.predict(X_cal)
        self._scores_cal = np.abs(y_cal - y_cal_hat)

        if verbose:
            print(f"[conformal] h_cal={best['h_cal']:.4f}  "
                  f"val_cov={best['coverage']:.4f}  "
                  f"val_width={best['width']:.4f}")
        return self

    def predict_interval_conformal(self, X, alpha=None):
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

        y_hat = self.predict(X)
        return y_hat - q_hat, y_hat + q_hat

    def predict_zone(self, X):
        return self.predict(X)