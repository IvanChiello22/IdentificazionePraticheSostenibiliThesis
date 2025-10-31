# T19 POS (energy/efficiency-aware checkpointing) — snippet 28
# T19 POS — NumPy: checkpoint leggero in .npz compresso
import numpy as np, os
os.makedirs('ckpt', exist_ok=True)
W = np.random.randn(64, 10); b = np.random.randn(10)

# Energy-aware lightweight pruning: prune a small fraction of weights,
# starting from the most energy-intensive output columns (highest column energy),
# removing the smallest-magnitude weights in those columns first.
sparsity_target = 0.10  # fraction of total weights to prune (10%)
total_weights = W.size
num_prune = int(total_weights * sparsity_target)
if num_prune > 0:
    # Column energy (sum of squared weights per output)
    col_energy = np.sum(W * W, axis=0)
    # Order columns by descending energy
    col_order = np.argsort(-col_energy)
    # Rank for each column: 0 = most energy-intensive
    col_rank = np.empty_like(col_order)
    col_rank[col_order] = np.arange(col_order.size)

    r, c = W.shape
    cols = np.tile(np.arange(c), r)
    abs_vals = np.abs(W).ravel()
    # lexsort keys: (abs_vals, col_rank[cols]) -> primary key is col_rank (most energy first),
    # secondary key is abs_vals (smallest magnitudes first)
    order_flat = np.lexsort((abs_vals, col_rank[cols]))
    prune_indices = order_flat[:num_prune]
    W.ravel()[prune_indices] = 0

# Use atomic saves for checkpoints to avoid corruption on premature termination
tmp_path = f'ckpt/.tmp_{os.getpid()}.npz'
final_best = 'ckpt/model_best.npz'
latest_ckpt = 'ckpt/latest.npz'

np.savez_compressed(tmp_path, W=W, b=b)
os.replace(tmp_path, final_best)

# Also keep a latest checkpoint (overwrites atomically)
tmp_latest = f'ckpt/.tmp_latest_{os.getpid()}.npz'
np.savez_compressed(tmp_latest, W=W, b=b)
os.replace(tmp_latest, latest_ckpt)

print('done')