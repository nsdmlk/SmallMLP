import numpy as np
from smallmlp import SmallMLPRegressor

rng = np.random.default_rng(0)
n, d = 200, 5
X = rng.normal(size=(n, d))
y = X[:, 0] * 2.0 - X[:, 1] + 0.3 * rng.normal(size=n)
# add outliers
y[:5] += 20.0

model = SmallMLPRegressor(h_init=1.0, max_iter=100, verbose=True)
model.fit(X, y)

X_test = rng.normal(size=(10, d))
y_hat = model.predict(X_test)
y_hat, delta = model.predict_zone(X_test)
lo, hi = model.predict_interval(X_test, alpha=0.95)

print("h =", model.get_h())
for i in range(5):
    print(f"y={y_hat[i]:+.3f}  zone=[{lo[i]:+.3f}, {hi[i]:+.3f}]  delta={delta[i]:.3f}")