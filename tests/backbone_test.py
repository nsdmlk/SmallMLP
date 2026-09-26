import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from smallmlp.nn import AdaptiveMLPRegressorHetero

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=300, random_state=0)

reg = AdaptiveMLPRegressorHetero(verbose=True)
reg.fit(X_tr, y_tr)

y_hat, y_std = reg.predict(X_te, return_std=True)

mae = np.abs(y_te - y_hat).mean()
print(f"MAE: {mae:.2f}")
print(f"Mean std: {y_std.mean():.2f}")
print(f"Std range: [{y_std.min():.2f}, {y_std.max():.2f}]")

# coverage of 1-sigma interval
in_1sigma = np.mean(np.abs(y_te - y_hat) <= y_std)
print(f"1-sigma coverage: {in_1sigma:.3f}  (target ~0.68)")
in_2sigma = np.mean(np.abs(y_te - y_hat) <= 2 * y_std)
print(f"2-sigma coverage: {in_2sigma:.3f}  (target ~0.95)")