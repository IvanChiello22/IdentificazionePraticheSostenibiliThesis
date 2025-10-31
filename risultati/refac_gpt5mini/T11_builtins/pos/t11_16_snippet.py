import numpy as np
A = np.random.randn(5000, 64)
B = np.random.randn(3000, 64)
D2 = np.square(A).sum(axis=1, keepdims=True) + np.square(B).sum(axis=1)[None, :] - 2 * A.dot(B.T)
np.maximum(D2, 0.0, out=D2)
np.sqrt(D2, out=D2)
D = D2
print(D.shape)