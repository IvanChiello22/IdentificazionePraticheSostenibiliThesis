# T19 POS (energy/efficiency-aware checkpointing) — snippet 28
# T19 POS — NumPy: checkpoint leggero in .npz compresso
import numpy as np, os
os.makedirs('ckpt', exist_ok=True)
W = np.random.randn(64, 10); b = np.random.randn(10)
np.savez_compressed('ckpt/model_best.npz', W=W, b=b)
print('done')
