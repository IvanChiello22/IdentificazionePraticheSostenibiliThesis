# T11 POS — snippet 8
import numpy as np
A = np.random.randn(5000, 64)
B = np.random.randn(3000, 64)
norms_A = np.sum(A * A, axis=1)[:, None]
norms_B = np.sum(B * B, axis=1)[None, :]
D2 = norms_A + norms_B - 2 * A @ B.T
D = np.sqrt(np.maximum(D2, 0.0))
print(D.shape)