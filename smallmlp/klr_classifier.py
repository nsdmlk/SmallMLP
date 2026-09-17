import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from .klr import klr_loo_bce_loss, klr_forward
from .conformal import weighted_quantile, _to_tensor
from .kernels import gaussian_weights, effective_sample_size


class KLRClassifier(BaseEstimator, ClassifierMixin):
    """Kernel Logistic Regression for small data.

    Point: sigmoid(alpha^T K(x, X_train) + b).
    alpha, b, h learned jointly by LOO-BCE + L2.
    """

    def __init__(self, h_min=0.01, h_max=10.0, lam=1e-3,
                 max_iter=30, inner_iter=20, tol=1e-6, verbose=False):
        self.h_min = h_min
        self.h_max = h_max
        self.lam = lam
        self.max_iter = max_iter
        self.inner_iter = inner_iter
        self.tol = tol
        self.verbose = verbose

    def _h_from_psi(self, psi):
        return self.h_min + (self.h_max - self.h_min) * torch.sigmoid(psi)

    @staticmethod
    def _psi_from_h(h, h_min, h_max):
        p = (h - h_min) / (h_max - h_min)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"KLRClassifier supports binary only, got {len(self.classes_)}"
            )
        y_bin = (y == self.classes_[1]).astype(np.float64)

        X = X.astype(np.float64)
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std == 0] = 1.0

        Xs = (X - self._x_mean) / self._x_std
        X_t = torch.tensor(Xs, dtype=torch.float64)
        y_t = torch.tensor(y_bin, dtype=torch.float64)
        n = X_t.shape[0]

        alpha = torch.nn.Parameter(torch.zeros(n, dtype=torch.float64))
        b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        psi0 = self._psi_from_h(1.0, self.h_min, self.h_max)
        psi = torch.nn.Parameter(
            torch.full((self.n_features_in_,), psi0, dtype=torch.float64)
        )

        optimizer = torch.optim.LBFGS(
            [alpha, b, psi],
            max_iter=self.inner_iter,
            tolerance_grad=self.tol,
            tolerance_change=self.tol,
            line_search_fn="strong_wolfe",
        )

        prev_loss = None
        for it in range(self.max_iter):
            def closure():
                optimizer.zero_grad()
                loss = klr_loo_bce_loss(
                    alpha, b, psi, X_t, y_t,
                    self.h_min, self.h_max, lam=self.lam,
                )
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            loss_val = float(loss.detach())
            if self.verbose:
                print(f"[KLR] iter {it:3d}  loss={loss_val:.6e}")
            if prev_loss is not None and abs(prev_loss - loss_val) < self.tol:
                break
            prev_loss = loss_val

        self._alpha = alpha.detach()
        self._b = b.detach()
        self._psi = psi.detach()
        self._X_train = X_t
        self._y_train = y_t
        self._loss_ = prev_loss
        return self

    def _prepare_query(self, X):
        check_is_fitted(self, ["_alpha", "_b", "_psi", "_X_train"])
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        Xs = (X - self._x_mean) / self._x_std
        return torch.tensor(Xs, dtype=torch.float64)

    def _predict_p(self, X):
        X_t = self._prepare_query(X)
        with torch.no_grad():
            h = self._h_from_psi(self._psi)
            p = klr_forward(X_t, self._X_train, self._alpha, self._b, h)
        return p.numpy()

    def predict_proba(self, X):
        p = self._predict_p(X)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        p = self._predict_p(X)
        return self.classes_[(p > 0.5).astype(int)]

    def get_h(self):
        check_is_fitted(self, ["_psi"])
        return self._h_from_psi(self._psi).numpy()

    # ---------------- conformal ----------------

    def fit_conformal(self, X_cal, y_cal, X_val, y_val,
                      alpha=0.1, h_grid=None, verbose=False):
        check_is_fitted(self, ["_alpha", "_b", "_psi", "_X_train"])

        y_cal_bin = (y_cal == self.classes_[1]).astype(np.float64)
        y_val_bin = (y_val == self.classes_[1]).astype(np.float64)

        # reuse classification tuning from conformal module
        from .conformal import tune_h_cal_classification
        best = tune_h_cal_classification(
            self, X_cal, y_cal_bin, X_val, y_val_bin,
            alpha=alpha, h_grid=h_grid,
        )
        self._h_cal = np.full(self.n_features_in_, best["h_cal"])
        self._alpha_conformal = alpha

        p_cal = self._predict_p(X_cal)
        self._scores_cal = np.where(y_cal_bin == 1, 1.0 - p_cal, p_cal)
        self._X_cal_raw = np.asarray(X_cal, dtype=np.float64)

        if verbose:
            print(f"[conformal] h_cal={best['h_cal']:.4f}  "
                  f"val_cov={best['coverage']:.4f}  "
                  f"val_size={best['size']:.4f}")
        return self

    def predict_set(self, X, alpha=None):
        check_is_fitted(self, ["_h_cal", "_scores_cal", "_X_cal_raw"])
        if alpha is None:
            alpha = self._alpha_conformal

        from .conformal import conformal_qhat_classification
        X = check_array(X, dtype=np.float64)
        X_t = _to_tensor(self, X)
        X_cal_t = _to_tensor(self, self._X_cal_raw)
        scores_t = torch.tensor(self._scores_cal, dtype=torch.float64)
        h_cal_t = torch.tensor(self._h_cal, dtype=torch.float64)

        q_hat = conformal_qhat_classification(
            X_t, X_cal_t, scores_t, h_cal_t, alpha
        )
        p_test = self._predict_p(X)

        sets = []
        for i in range(len(X)):
            s_0 = p_test[i]
            s_1 = 1.0 - p_test[i]
            c = set()
            if s_0 <= q_hat[i]:
                c.add(0)
            if s_1 <= q_hat[i]:
                c.add(1)
            if len(c) == 0:
                c.add(0 if s_0 < s_1 else 1)
            sets.append(c)
        return sets