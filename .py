from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from smallmlp import SmallMLPClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, train_size=200, stratify=y, random_state=0
)

clf = SmallMLPClassifier(verbose=True)
clf.fit(X_tr, y_tr)
p = clf.predict_proba(X_te)[:, 1]
print("AUC:", roc_auc_score(y_te, p))
print("h:", clf.get_h())
print("acc:", (clf.predict(X_te) == y_te).mean())