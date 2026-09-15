import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .core import forward
from .loss import loo_huber_loss
from .conformal import conformal_qhat, tune_h_cal, _to_tensor


class SmallMLPRegressor(BaseEstimator, RegressorMixin):
    """Small-data regressor: learned-bandwidth Nadaraya-Watson + prediction zone.

    Two interval methods:
      - predict_interval: heuristic Gaussian zone  [y_hat ± z * delta]
      - predict_interval_conformal: weighted conformal with finite-sample
        coverage guarantee
    """

    def __init__(self, h_min=0.01, h_max=10.0, max_iter=30, inner_iter=10,
                 tol=1e-6, verbose=False, alpha=1e-3):
        self.h_min = h_min
        self.h_max = h_max
        self.max_iter = max_iter
        self.inner_iter = inner_iter
        self.tol = tol
        self.verbose = verbose
        self.alpha = alpha

    # ---------------- bandwidth parametrization ----------------

    def _h_from_psi(self, psi):
        return self.h_min + (self.h_max - self.h_min) * torch.sigmoid(psi)

    @staticmethod
    def _psi_from_h(h, h_min, h_max):
        p = (h - h_min) / (h_max - h_min)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    # ---------------- fit (point predictor) ----------------

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std == 0] = 1.0
        self._y_mean = float(y.mean())
        y_std = float(y.std())
        self._y_std = y_std if y_std > 0 else 1.0

        Xs = (X - self._x_mean) / self._x_std
        ys = (y - self._y_mean) / self._y_std

        X_t = torch.tensor(Xs, dtype=torch.float64)
        y_t = torch.tensor(ys, dtype=torch.float64)

        psi0 = self._psi_from_h(1.0, self.h_min, self.h_max)
        psi = torch.nn.Parameter(
            torch.full((self.n_features_in_,), psi0, dtype=torch.float64)
        )

        optimizer = torch.optim.LBFGS(
            [psi],
            max_iter=self.inner_iter,
            tolerance_grad=self.tol,
            tolerance_change=self.tol,
            line_search_fn="strong_wolfe",
        )

        prev_loss = None
        for it in range(self.max_iter):
            def closure():
                optimizer.zero_grad()
                h = self._h_from_psi(psi)
                loss = loo_huber_loss(h, X_t, y_t, forward)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            loss_val = float(loss.detach())

            if self.verbose:
                print(f"[SmallMLP] iter {it:3d}  loss={loss_val:.6e}")

            if prev_loss is not None and abs(prev_loss - loss_val) < self.tol:
                break
            prev_loss = loss_val

        self._psi = psi.detach()
        self._X_train = X_t
        self._y_train = y_t
        self._loss_ = prev_loss
        return self

    # ---------------- point prediction ----------------

    def _prepare_query(self, X):
        check_is_fitted(self, ["_psi", "_X_train", "_y_train"])
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        Xs = (X - self._x_mean) / self._x_std
        return torch.tensor(Xs, dtype=torch.float64)

    def predict(self, X):
        X_t = self._prepare_query(X)
        with torch.no_grad():
            h = self._h_from_psi(self._psi)
            y_hat, _, _ = forward(X_t, self._X_train, self._y_train, h,
                                  alpha=self.alpha)
        return y_hat.numpy() * self._y_std + self._y_mean

    # ---------------- heuristic zone ----------------

    def predict_zone(self, X):
        X_t = self._prepare_query(X)
        with torch.no_grad():
            h = self._h_from_psi(self._psi)
            y_hat, delta, _ = forward(X_t, self._X_train, self._y_train, h,
                                      alpha=self.alpha)
        return (
            y_hat.numpy() * self._y_std + self._y_mean,
            delta.numpy() * self._y_std,
        )

    def predict_interval(self, X, alpha=0.95):
        from scipy.stats import norm
        k = norm.ppf((1.0 + alpha) / 2.0)
        y_hat, delta = self.predict_zone(X)
        return y_hat - k * delta, y_hat + k * delta

    # ---------------- weighted conformal ----------------

    def fit_conformal(self, X_cal, y_cal, X_val, y_val,
                      alpha=0.05, h_grid=None, lambda_penalty=10.0,
                      verbose=False):
        """Tune h_cal on validation, store calibration residuals.

        Call after fit(). X_cal/X_val must be disjoint from training data.
        """
        check_is_fitted(self, ["_psi", "_X_train", "_y_train"])

        best = tune_h_cal(
            self, X_cal, y_cal, X_cal, y_cal, X_val, y_val,
            alpha=alpha, h_grid=h_grid, lambda_penalty=lambda_penalty,
        )
        self._h_cal = np.full(self.n_features_in_, best["h_cal"])
        self._alpha_conformal = alpha

        # calibration residuals in ORIGINAL y units
        y_cal_hat = self.predict(X_cal)
        self._residuals_cal = np.abs(y_cal - y_cal_hat)
        self._X_cal_raw = np.asarray(X_cal, dtype=np.float64)

        if verbose:
            print(f"[conformal] h_cal={best['h_cal']:.4f}  "
                  f"val_cov={best['coverage']:.4f}  "
                  f"val_width={best['width']:.4f}")

        return self

    def predict_interval_conformal(self, X, alpha=None):
        """Weighted conformal interval using stored calibration residuals.

        Returns (lower, upper) in original y units.
        """
        check_is_fitted(
            self, ["_h_cal", "_residuals_cal", "_X_cal_raw"]
        )
        if alpha is None:
            alpha = self._alpha_conformal

        X = check_array(X, dtype=np.float64)
        X_t = _to_tensor(self, X)
        X_cal_t = _to_tensor(self, self._X_cal_raw)

        residuals_t = torch.tensor(
            self._residuals_cal, dtype=torch.float64
        )
        h_cal_t = torch.tensor(self._h_cal, dtype=torch.float64)

        q_hat = conformal_qhat(
            X_t, X_cal_t, residuals_t, h_cal_t, alpha
        )
        y_hat = self.predict(X)
        return y_hat - q_hat, y_hat + q_hat

    # ---------------- diagnostics ----------------

    def get_h(self):
        check_is_fitted(self, ["_psi"])
        return self._h_from_psi(self._psi).numpy()

    def get_h_cal(self):
        check_is_fitted(self, ["_h_cal"])
        return self._h_cal.copy()