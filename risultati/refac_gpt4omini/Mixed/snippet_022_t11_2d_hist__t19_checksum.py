import numpy as np
import torch
import torch.nn as nn
import io
import hashlib
import os
import json

# T11 — NumPy: 2D histogram + vectorized normalization
x = np.random.randn(100000)
y = 0.5 * x + np.random.randn(100000) * 0.5
H, xedges, yedges = np.histogram2d(x, y, bins=64, range=[[-4, 4], [-4, 4]])
P = H / H.sum()
print(P.shape, round(P.max(), 4))

# T19 — Deduplicate checkpoints by SHA256 (skip identical content)
model = nn.Sequential(nn.Linear(12, 12), nn.ReLU(), nn.Linear(12, 2))
os.makedirs('ckpt', exist_ok=True)
buffer = io.BytesIO()
torch.save(model.state_dict(), buffer)
data = buffer.getvalue()
hash_value = hashlib.sha256(data).hexdigest()
index = {'sha256': hash_value, 'file': f'ckpt/{hash_value}.pt'}
if not os.path.exists(index['file']):
    with open(index['file'], 'wb') as f:
        f.write(data)
with open('ckpt/index.json', 'w') as f:
    json.dump(index, f)
print('done', hash_value[:8])