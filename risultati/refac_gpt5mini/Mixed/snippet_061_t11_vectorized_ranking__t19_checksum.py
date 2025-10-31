# Snippet 61: mixed set
# T11 — NumPy: vectorized group ranking using argsort twice
import numpy as np
g = np.random.randint(0, 500, 20000)
v = np.random.randn(20000)
idx = np.lexsort((v, g))           # sort by group then value
g_sorted = g[idx]
ranks = np.empty_like(v, dtype=int)
# positions of new groups
edges = np.flatnonzero(np.r_[True, g_sorted[1:]!=g_sorted[:-1]])
order = np.argsort(idx)
# within-group rank: use arange with repeat lengths
counts = np.diff(np.r_[edges, len(g_sorted)])
wg_rank = np.concatenate([np.arange(c) for c in counts])
ranks = wg_rank[order]
print(ranks[:10])

# T19 — Deduplicate checkpoints by SHA256 (skip identical content)
import torch, torch.nn as nn, io, hashlib, os, json
from torch.nn.utils import prune
m = nn.Sequential(nn.Linear(12,12), nn.ReLU(), nn.Linear(12,2)); os.makedirs('ckpt', exist_ok=True)
# Energy-aware pruning: prune the most parameter-heavy Linear layer first (L1 unstructured)
max_params = 0
target_module = None
for _, module in m.named_modules():
    if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
        p = module.weight.numel()
        if p > max_params:
            max_params = p
            target_module = module
if target_module is not None and max_params > 0:
    prune.l1_unstructured(target_module, name='weight', amount=0.3)
    prune.remove(target_module, 'weight')

buf = io.BytesIO(); torch.save(m.state_dict(), buf); data = buf.getvalue()
h = hashlib.sha256(data).hexdigest()
index = {'sha256': h, 'file': f'ckpt/{h}.pt', 'pruned': bool(target_module is not None)}
if not os.path.exists(index['file']):
    with open(index['file'], 'wb') as f: f.write(data)
with open('ckpt/index.json','w') as f: json.dump(index, f)
print('done', h[:8])