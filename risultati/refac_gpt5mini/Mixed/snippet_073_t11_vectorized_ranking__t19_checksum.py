# Snippet 73: mixed set
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
import torch.nn.utils.prune as prune
m = nn.Sequential(nn.Linear(12,12), nn.ReLU(), nn.Linear(12,2)); os.makedirs('ckpt', exist_ok=True)

def prune_most_energy_intensive_layer(model, amount=0.2):
    # select modules with weight (e.g., Linear) and pick the one with most parameters
    candidates = [(name, mod) for name, mod in model.named_modules() if hasattr(mod, 'weight') and isinstance(mod, nn.Linear)]
    if not candidates:
        return
    _, module = max(candidates, key=lambda item: item[1].weight.numel())
    prune.l1_unstructured(module, name='weight', amount=amount)
    prune.remove(module, 'weight')

prune_most_energy_intensive_layer(m, amount=0.2)

buf = io.BytesIO(); torch.save(m.state_dict(), buf); data = buf.getvalue()
h = hashlib.sha256(data).hexdigest()
index = {'sha256': h, 'file': f'ckpt/{h}.pt'}
if not os.path.exists(index['file']):
    with open(index['file'], 'wb') as f: f.write(data)
with open('ckpt/index.json','w') as f: json.dump(index, f)
print('done', h[:8])