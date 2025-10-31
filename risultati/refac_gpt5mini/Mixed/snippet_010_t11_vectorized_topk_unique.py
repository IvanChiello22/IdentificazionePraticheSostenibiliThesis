# Snippet 10: t11_vectorized_topk_unique
# T11 — NumPy: per‑row top‑k without loops via argpartition
import numpy as np

def _compute_topk_rows(matrix, k):
    """
    Compute per-row top-k values and intermediate indices using NumPy
    without changing the original semantics.
    Returns: (idx, vals, sorted_idx, topk)
    """
    idx = np.argpartition(-matrix, k, axis=1)[:, :k]
    vals = np.take_along_axis(matrix, idx, axis=1)
    sorted_idx = np.argsort(-vals, axis=1)
    topk = np.take_along_axis(vals, sorted_idx, axis=1)
    return idx, vals, sorted_idx, topk

X = np.random.randn(200, 1000)
k = 7
idx, vals, sorted_idx, topk = _compute_topk_rows(X, k)
print(topk.shape, round(topk.mean(), 4))