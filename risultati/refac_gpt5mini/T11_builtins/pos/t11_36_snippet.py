import numpy as np
A=np.random.randn(5000,64);B=np.random.randn(3000,64)
n_rows_A, n_rows_B = A.shape[0], B.shape[0]
D = np.empty((n_rows_A, n_rows_B), dtype=A.dtype)
block_cols = 500
for j0 in range(0, n_rows_B, block_cols):
    j1 = min(j0 + block_cols, n_rows_B)
    B_chunk = B[j0:j1]
    D2_chunk = (A*A).sum(1, keepdims=True) + (B_chunk*B_chunk).sum(1)[None, :] - 2*A@B_chunk.T
    D_chunk = np.sqrt(np.maximum(D2_chunk, 0.0))
    D[:, j0:j1] = D_chunk
    np.save(f'D_checkpoint_{j0}.npy', D[:, j0:j1])
D2 = D2_chunk
print(D.shape)