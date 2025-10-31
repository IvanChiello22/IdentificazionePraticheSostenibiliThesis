# T11 POS — snippet 24
import numpy as np
A = np.random.randn(5000, 64).astype(np.float32)
B = np.random.randn(3000, 64).astype(np.float32)
D2 = (A * A).sum(1, keepdims=True) + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
D = np.sqrt(np.maximum(D2, np.float32(0.0)))
print(D.shape)