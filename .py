import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from smallmlp import SmallMLPClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, train_size=200, stratify=y, random_state=0
)

for method in ["nw", "klr"]:
    clf = SmallMLPClassifier(
        point_method=method,
        class_weight="balanced",
        verbose=False,
    )
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, 1]
    print(f"{method:4s}  AUC={roc_auc_score(y_te, p):.4f}  h={clf.get_h()[:3]}")