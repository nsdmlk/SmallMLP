import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from smallmlp import SmallMLPClassifier

for name, uid in [("vowel", 59), ("ecoli", 39), ("vehicle", 149), ("heart_statlog", 145), ("haberman", 43)]:
    try:
        ds = fetch_ucirepo(id=uid)
        X = ds.data.features.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        y = LabelEncoder().fit_transform(ds.data.targets.iloc[:, 0].astype(str))
        mask = np.isfinite(X).all(axis=1)
        X, y = X[mask], y[mask]
        if len(y) > 500:
            idx = np.random.default_rng(0).choice(len(y), size=500, replace=False)
            X, y = X[idx], y[idx]
        print(f"\n{name}  n={len(y)}, d={X.shape[1]}, K={len(np.unique(y))}")
        for seed in [0, 1, 42]:
            for mode in ["classic", "log_sqrt"]:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                accs = []
                for tr, te in skf.split(X, y):
                    clf = SmallMLPClassifier(
                        class_weight="balanced",
                        max_epochs=1000, patience=50,
                        width_mode=mode, random_state=seed,
                    )
                    clf.fit(X[tr], y[tr])
                    accs.append(accuracy_score(y[te], clf.predict(X[te])))
                print(f"  seed={seed:3d}  {mode:12s}  "
                      f"mean={np.mean(accs):.4f}  min={np.min(accs):.3f}  max={np.max(accs):.3f}")
    except Exception as e:
        print(f"{name}: FAILED - {e}")