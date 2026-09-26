import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from smallmlp import SmallMLPRegressor

X, y = load_diabetes(return_X_y=True)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0)
X_cal, X_val, y_cal, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

reg = SmallMLPRegressor(verbose=True)
reg.fit(X_tr, y_tr)
y_hat = reg.predict(X_val)
print(f"MAE: {mean_absolute_error(y_val, y_hat):.2f}")

reg.fit_conformal(X_cal, y_cal, X_val, y_val, alpha=0.1, verbose=True)
lo, hi = reg.predict_interval_conformal(X_val, alpha=0.1)
cov = np.mean((y_val >= lo) & (y_val <= hi))
print(f"Coverage: {cov:.3f}  Width: {np.mean(hi-lo):.2f}")