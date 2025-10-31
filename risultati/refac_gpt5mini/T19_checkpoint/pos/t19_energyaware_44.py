import os
import numpy as np

os.makedirs('ckpt', exist_ok=True)
W = np.random.randn(64, 10); b = np.random.randn(10)
_tmp_path = os.path.join('ckpt', 'model_best.npz.tmp')
_final_path = os.path.join('ckpt', 'model_best.npz')
np.savez_compressed(_tmp_path, W=W, b=b)
os.replace(_tmp_path, _final_path)
print('done')