import numpy as np
A = np.random.randn(5000, 64)
B = np.random.randn(3000, 64)
sqA = np.sum(A * A, axis=1, keepdims=True)
sqB = np.sum(B * B, axis=1)[None, :]
D2 = sqA + sqB - 2 * (A @ B.T)
D = np.sqrt(np.clip(D2, 0.0, None))
print(D.shape)