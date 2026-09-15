from smallmlp.conformal import weighted_quantile
import numpy as np

# симуляция: residuals = [40, 45, 50, ..., 150]
res = np.abs(np.random.randn(88) * 50 + 40)
w = np.ones(88)
q = weighted_quantile(res, w, 0.92)
print("q_hat =", q, "  (должно быть ~90-й процентиль)")
print("90th percentile:", np.percentile(res, 92))
print("Coverage if test residuals same:", np.mean(res <= q))