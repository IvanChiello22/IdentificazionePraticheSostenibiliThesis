# Snippet 6: t11_vectorized_ranking
# T11 — NumPy: vectorized group ranking using argsort twice
import numpy as np
g = np.random.randint(0, 500, 20000)
v = np.random.randn(20000)
idx = np.lexsort((v, g))           # sort by group then value
g_sorted = g[idx]
ranks = np.empty_like(v, dtype=int)
# positions of new groups — compute group counts using built-in unique
uniq_vals, counts = np.unique(g_sorted, return_counts=True)
edges = np.r_[0, np.cumsum(counts)[:-1]]
order = np.argsort(idx)
# within-group rank: use arange with repeat lengths
wg_rank = np.concatenate([np.arange(c) for c in counts])
ranks = wg_rank[order]
print(ranks[:10])