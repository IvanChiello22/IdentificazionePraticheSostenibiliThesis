import numpy as np
A = np.random.randn(5000, 64)
B = np.random.randn(3000, 64)
D2 = np.sum(A**2, axis=1, keepdims=True) + np.sum(B**2, axis=1)[None, :] - 2 * np.dot(A, B.T)
D = np.sqrt(np.maximum(D2, 0.0))
print(D.shape)