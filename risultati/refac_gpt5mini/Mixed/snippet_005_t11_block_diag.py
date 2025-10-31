# Snippet 5: t11_block_diag
# T11 — SciPy: block diagonal assembly + sparse matmul
import numpy as np
from scipy.sparse import block_diag, random as sprand
blocks = [sprand(200,200, density=0.01, format='csr') for _ in range(8)]
A = block_diag(blocks, format='csr')
x = np.random.randn(A.shape[1])
y = A @ x
print(A.shape, A.nnz, y.shape)