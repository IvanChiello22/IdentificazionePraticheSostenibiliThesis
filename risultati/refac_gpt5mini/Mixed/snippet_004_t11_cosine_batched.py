import numpy as np
A = np.random.randn(500, 128)
B = np.random.randn(600, 128)
_eps = 1e-9

def compute_cosine_similarity(A, B, eps=_eps):
    norms_A = np.linalg.norm(A, axis=1, keepdims=True)
    norms_B = np.linalg.norm(B, axis=1, keepdims=True)
    # avoid division by very small numbers
    np.maximum(norms_A, eps, out=norms_A)
    np.maximum(norms_B, eps, out=norms_B)
    # normalize in-place to avoid extra memory allocation
    np.divide(A, norms_A, out=A)
    np.divide(B, norms_B, out=B)
    return A @ B.T

S = compute_cosine_similarity(A, B)
print(S.shape, round(S.max(), 4))