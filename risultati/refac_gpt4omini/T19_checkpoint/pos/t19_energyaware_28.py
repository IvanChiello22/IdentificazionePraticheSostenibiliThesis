import numpy as np
import os

os.makedirs('ckpt', exist_ok=True)
W = np.random.randn(64, 10)
b = np.random.randn(10)
np.savez_compressed('ckpt/model_best.npz', W=W, b=b)
print('done')