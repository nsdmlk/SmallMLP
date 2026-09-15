from sklearn.datasets import load_diabetes
from smallmlp import SmallMLPRegressor
d = load_diabetes()
m = SmallMLPRegressor(verbose=True)
m.fit(d.data, d.target)
print("h:", m.get_h())
print("loss:", m._loss_)