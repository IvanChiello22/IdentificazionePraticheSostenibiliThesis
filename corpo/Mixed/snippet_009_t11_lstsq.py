# Snippet 9: t11_lstsq
# T11 — NumPy/Scipy: batched least squares via lstsq / pinv
import numpy as np
A = np.random.randn(100, 40)
b = np.random.randn(100, 5)
x, *_ = np.linalg.lstsq(A, b, rcond=None)
residual = np.linalg.norm(A@x - b)
print(x.shape, round(residual,4))
