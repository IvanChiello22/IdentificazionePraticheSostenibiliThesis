# Snippet 50: triple mix
# T11 — NumPy: per‑row top‑k without loops via argpartition
import numpy as np
X = np.random.randn(200, 1000)
k = 7
idx = np.argpartition(-X, k, axis=1)[:, :k]
vals = np.take_along_axis(X, idx, axis=1)
sorted_idx = np.argsort(-vals, axis=1)
topk = np.take_along_axis(vals, sorted_idx, axis=1)
print(topk.shape, round(topk.mean(),4))

# T15 — Energy‑aware pruning for MLP neurons (cost ~ fan_in + activation size)
import torch, torch.nn as nn, torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self,d=128,h=256):
        super().__init__()
        self.d = d
        self.fc1 = nn.Linear(d, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, d)
    def forward(self,x):
        return self.fc2(self.act(self.fc1(x)))

def prune_hidden_energy(fc1: nn.Linear, keep_ratio, batch_tokens=2048):
    with torch.no_grad():
        orig_out = fc1.out_features
        # neuron importance (mean abs weight)
        imp = fc1.weight.detach().abs().mean(dim=1)
        energy = fc1.in_features + batch_tokens  # proxy
        score = imp / (energy + 1e-9)
        k = max(1, int(keep_ratio * orig_out))
        keep = torch.topk(score, k).indices
        # prune weights and bias
        fc1.weight = nn.Parameter(fc1.weight[keep].clone())
        if fc1.bias is not None:
            fc1.bias = nn.Parameter(fc1.bias[keep].clone())
        fc1.out_features = k
        return keep

m = MLP()
keep_idx = prune_hidden_energy(m.fc1, 0.5, 4096)
# adjust following linear layer to match pruned hidden size when applicable
if hasattr(m, 'fc2') and isinstance(m.fc2, nn.Linear):
    with torch.no_grad():
        m.fc2.weight = nn.Parameter(m.fc2.weight[:, keep_idx].clone())
        m.fc2.in_features = m.fc1.out_features

x = torch.randn(8, m.d)
print(m(x).shape, m.fc1.out_features)

# T19 — Threaded async save using concurrent.futures with atomic checkpoint writes
import os
from concurrent.futures import ThreadPoolExecutor
import tempfile

os.makedirs('ckpt', exist_ok=True)
m = nn.Sequential(nn.Linear(16,16), nn.ReLU(), nn.Linear(16,2))
def save(path):
    # atomic save via temporary file + replace to avoid partial writes (checkpoint safety)
    dirpath = os.path.dirname(path) or '.'
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, prefix=os.path.basename(path), suffix='.tmp')
    os.close(fd)
    try:
        torch.save({'state': m.state_dict()}, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

with ThreadPoolExecutor(max_workers=2) as ex:
    futs = [ex.submit(save, f'ckpt/shard_{i}.pt') for i in range(3)]
    _ = [f.result() for f in futs]
print('done')