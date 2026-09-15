import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from .core import forward
from .loss import loo_bce_loss


class SmallMLPClassifier(BaseEstimator, ClassifierMixin):
    """Small-data binary classifier: weighted soft-vote + zone.

    predict_proba returns [1-p, p]. predict returns thresholded labels.
    predict_zone returns (p_hat, delta) for uncertainty analysis.
    """

    def __init__(self, h_min=0.01, h_max=10.0, max_iter=30, inner_iter=10,
                 tol=1e-6, verbose=False):
        self.h_min = h_min
        self.h_max = h_max
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

        # encode labels to {0, 1}
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"SmallMLPClassifier supports binary classification only, "
                f"got {len(self.classes_)} classes."
            )
        y_bin = (y == self.classes_[1]).astype(np.float64)

        X = X.astype(np.float64)
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std == 0] = 1.0

        Xs = (X - self._x_mean) / self._x_std
        X_t = torch.tensor(Xs, dtype=torch.float64)
        y_t = torch.tensor(y_bin, dtype=torch.float64)

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
                loss = loo_bce_loss(h, X_t, y_t, forward)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            loss_val = float(loss.detach())

            if self.verbose:
                print(f"[SmallMLPClassifier] iter {it:3d}  loss={loss_val:.6e}")

            if prev_loss is not None and abs(prev_loss - loss_val) < self.tol:
                break
            prev_loss = loss_val

        self._psi = psi.detach()
        self._X_train = X_t
        self._y_train = y_t
        self._loss_ = prev_loss
        return self

    def _prepare_query(self, X):
        check_is_fitted(self, ["_psi", "_X_train", "_y_train"])
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        Xs = (X - self._x_mean) / self._x_std
        return torch.tensor(Xs, dtype=torch.float64)

    def _predict_p_raw(self, X):
        X_t = self._prepare_query(X)
        with torch.no_grad():
            h = self._h_from_psi(self._psi)
            p_hat, delta, _ = forward(X_t, self._X_train, self._y_train, h)
        return p_hat.numpy(), delta.numpy()

    def predict_proba(self, X):
        p, _ = self._predict_p_raw(X)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        p, _ = self._predict_p_raw(X)
        idx = (p > 0.5).astype(int)
        return self.classes_[idx]

    def predict_zone(self, X):
        """Return (p_hat, delta) in probability space."""
        return self._predict_p_raw(X)

    def get_h(self):
        check_is_fitted(self, ["_psi"])
        return self._h_from_psi(self._psi).numpy()