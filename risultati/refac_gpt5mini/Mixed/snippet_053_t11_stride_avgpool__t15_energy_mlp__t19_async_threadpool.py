# Snippet 53: triple mix
# T11 — NumPy: sliding-window average via as_strided (no Python loops)
import numpy as np
x = np.random.randn(512, 512)
k = 5
shape = (x.shape[0]-k+1, x.shape[1]-k+1, k, k)
strides = (x.strides[0], x.strides[1], x.strides[0], x.strides[1])
windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
avg = windows.mean(axis=(2,3))
print(avg.shape, float(avg.std()))

# T15 — Energy‑aware pruning for MLP neurons (cost ~ fan_in + activation size)
import torch, torch.nn as nn, torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self,d=128,h=256): super().__init__(); self.d=d; self.fc1=nn.Linear(d,h); self.fc2=nn.Linear(h,d)
    def forward(self,x): return self.fc2(F.gelu(self.fc1(x)))
def prune_hidden_energy(fc1: nn.Linear, keep_ratio, batch_tokens=2048):
    with torch.no_grad():
        imp = fc1.weight.detach().abs().mean(dim=1)  # neuron importance
        energy = fc1.in_features + batch_tokens  # proxy
        score = imp/(energy+1e-9); k=max(1,int(keep_ratio*fc1.out_features))
        keep = torch.topk(score,k).indices
        fc1.weight = nn.Parameter(fc1.weight[keep].clone())
        if fc1.bias is not None: fc1.bias = nn.Parameter(fc1.bias[keep].clone())
        fc1.out_features = k
m = MLP(); prune_hidden_energy(m.fc1, 0.5, 4096)
x = torch.randn(8, m.d); print(m(x).shape, m.fc1.out_features)

# T19 — Threaded async save using concurrent.futures
import torch, torch.nn as nn, os
from concurrent.futures import ThreadPoolExecutor
os.makedirs('ckpt', exist_ok=True)
m = nn.Sequential(nn.Linear(16,16), nn.ReLU(), nn.Linear(16,2))
def save(path): torch.save({'state': m.state_dict()}, path)
with ThreadPoolExecutor(max_workers=2) as ex:
    futs = [ex.submit(save, f'ckpt/shard_{i}.pt') for i in range(3)]
    _ = [f.result() for f in futs]
print('done')