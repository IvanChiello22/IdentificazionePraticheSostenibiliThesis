import numpy as np
A=np.random.randn(5000,64);B=np.random.randn(3000,64)
sqA = (A*A).sum(axis=1, keepdims=True)
sqB = (B*B).sum(axis=1)[None,:]
cross = A @ B.T
D2 = sqA + sqB - 2*cross
D = np.sqrt(np.clip(D2, 0.0, None))
print(D.shape)