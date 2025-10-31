import numpy as np
import torch
import torch.nn as nn
import io
import hashlib
import os
import json

# Snippet 73: mixed set
g = np.random.randint(0, 500, 20000)
v = np.random.randn(20000)
idx = np.lexsort((v, g))           # sort by group then value
g_sorted = g[idx]
ranks = np.empty_like(v, dtype=int)
# positions of new groups
edges = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
order = np.argsort(idx)
# within-group rank: use arange with repeat lengths
counts = np.diff(np.r_[edges, len(g_sorted)])
wg_rank = np.concatenate([np.arange(c) for c in counts])
ranks = wg_rank[order]
print(ranks[:10])

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