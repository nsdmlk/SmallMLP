import time
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from smallmlp import SmallMLPRegressor


def measure(model, X_train, y_train, X_test, n_repeats=3):
    fit_times, predict_times = [], []
    for _ in range(n_repeats):
        m = model  # already cloned by caller if needed
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        fit_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        m.predict(X_test)
        predict_times.append(time.perf_counter() - t0)
    return np.median(fit_times), np.median(predict_times)


def main():
    rng = np.random.default_rng(0)
    sizes = [(100, 5), (300, 10), (500, 20), (1000, 20), (2000, 20)]
    n_test = 200

    print(f"{'n':>6} {'d':>4} | {'model':<14} | {'fit (s)':>10} {'predict (s)':>12}")
    print("-" * 60)

    for n, d in sizes:
        X = rng.normal(size=(n, d))
        y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=n)
        X_test = rng.normal(size=(n_test, d))

        models = {
            "SmallMLP": lambda: SmallMLPRegressor(
                h_init=1.0, max_iter=100, tol=1e-8
            ),
            "MLP_default": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)),
            ]),
            "KNN_k5": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=5)),
            ]),
        }

        for name, factory in models.items():
            try:
                fit_t, pred_t = measure(factory(), X, y, X_test)
                print(f"{n:>6} {d:>4} | {name:<14} | {fit_t:>10.4f} {pred_t:>12.6f}")
            except Exception as e:
                print(f"{n:>6} {d:>4} | {name:<14} | FAILED: {e}")
        print("-" * 60)


if __name__ == "__main__":
    main()