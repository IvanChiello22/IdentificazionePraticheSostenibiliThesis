import numpy as np
import os

os.makedirs('ckpt', exist_ok=True)
weights = np.random.randn(64, 10)
bias = np.random.randn(10)
np.savez_compressed('ckpt/model_best.npz', weights=weights, bias=bias)
print('done')