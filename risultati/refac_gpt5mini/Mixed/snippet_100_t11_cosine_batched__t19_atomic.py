# Snippet 100: mixed set
# T11 — NumPy: cosine similarity, fully vectorized (no sklearn)
import numpy as np
A = np.random.randn(500, 128)
B = np.random.randn(600, 128)
A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
S = A @ B.T
print(S.shape, round(S.max(),4))

# T19 — Atomic checkpoint writes (write to tmp then rename), avoid partial files
import torch
import torch.nn as nn
from torch.nn.utils import prune
import os, tempfile, shutil
m = nn.Sequential(nn.Linear(20,40), nn.ReLU(), nn.Linear(40,3)); os.makedirs('ckpt', exist_ok=True)

# Energy-aware pruning: prune the most energy intensive layer first (here the first Linear has most params)
prune.l1_unstructured(m[0], name='weight', amount=0.5)
prune.remove(m[0], 'weight')  # make pruning permanent (keeps zeros)

state = {'model': m.state_dict(), 'meta': {'epoch': 1}}

def save_checkpoint(state, path):
    dirpath = os.path.dirname(path) or '.'
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dirpath, prefix='tmp_', suffix='.pt')
    os.close(fd)
    try:
        torch.save(state, tmp)
        shutil.move(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

save_checkpoint(state, 'ckpt/model.pt')
print('done')