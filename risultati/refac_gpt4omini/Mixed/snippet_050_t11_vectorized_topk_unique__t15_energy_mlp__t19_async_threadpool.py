import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from concurrent.futures import ThreadPoolExecutor

# Snippet 50: triple mix
X = np.random.randn(200, 1000)
k = 7
idx = np.argpartition(-X, k, axis=1)[:, :k]
vals = np.take_along_axis(X, idx, axis=1)
sorted_idx = np.argsort(-vals, axis=1)
topk = np.take_along_axis(vals, sorted_idx, axis=1)
print(topk.shape, round(topk.mean(), 4))

# T15 — Energy‑aware pruning for MLP neurons (cost ~ fan_in + activation size)
class MLP(nn.Module):
    def __init__(self, d=128, h=256):
        super().__init__()
        self.d = d
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

def prune_hidden_energy(fc1: nn.Linear, keep_ratio, batch_tokens=2048):
    with torch.no_grad():
        imp = fc1.weight.detach().abs().mean(dim=1)  # neuron importance
        energy = fc1.in_features + batch_tokens  # proxy
        score = imp / (energy + 1e-9)
        k = max(1, int(keep_ratio * fc1.out_features))
        keep = torch.topk(score, k).indices
        fc1.weight = nn.Parameter(fc1.weight[keep].clone())
        if fc1.bias is not None:
            fc1.bias = nn.Parameter(fc1.bias[keep].clone())
        fc1.out_features = k

m = MLP()
prune_hidden_energy(m.fc1, 0.5, 4096)
x = torch.randn(8, m.d)
print(m(x).shape, m.fc1.out_features)

# T19 — Threaded async save using concurrent.futures
os.makedirs('ckpt', exist_ok=True)
m = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 2))

def save(path):
    torch.save({'state': m.state_dict()}, path)

with ThreadPoolExecutor(max_workers=2) as ex:
    futs = [ex.submit(save, f'ckpt/shard_{i}.pt') for i in range(3)]
    _ = [f.result() for f in futs]

print('done')