# Snippet 44: triple mix
# T11 — NumPy: cosine similarity, fully vectorized (no sklearn)
import numpy as np
A = np.random.randn(500, 128)
B = np.random.randn(600, 128)
A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
S = A @ B.T
print(S.shape, round(S.max(),4))

# T15 — Energy‑aware global conv pruning using forward‑hook activation stats
import torch, torch.nn as nn, torch.nn.functional as F
stats = {}
def hook(name):
    def fn(_, __, out):
        B,C,H,W = out.shape
        stats[name] = stats.get(name, 0) + (H*W*C)
    return fn
class Net(nn.Module):
    def __init__(self): super().__init__(); self.c1=nn.Conv2d(3,20,3,padding=1); self.c2=nn.Conv2d(20,40,3,padding=1); self.c3=nn.Conv2d(40,80,3,padding=1)
    def forward(self,x): return self.c3(F.relu(self.c2(F.relu(self.c1(x)))))
net = Net(); net.c1.register_forward_hook(hook('c1')); net.c2.register_forward_hook(hook('c2')); net.c3.register_forward_hook(hook('c3'))
x = torch.randn(4,3,64,64); _ = net(x)
# map energy stats to module objects for correct attribution
energy = {net.c1: stats.get('c1', 0), net.c2: stats.get('c2', 0), net.c3: stats.get('c3', 0)}
def prune_out(conv, keep_ratio):
    with torch.no_grad():
        imp = conv.weight.detach().abs().view(conv.out_channels, -1).mean(1)
        e = torch.tensor(float(energy.get(conv, 1.0)))
        score = imp/(e+1e-9); k=max(1,int(keep_ratio*conv.out_channels))
        keep = torch.topk(score,k).indices
        conv.weight = nn.Parameter(conv.weight[keep].clone()); conv.out_channels = k
# prune starting from the most energy intensive layer
convs = [net.c1, net.c2, net.c3]
sorted_convs = sorted(convs, key=lambda m: energy.get(m, 0), reverse=True)
for conv in sorted_convs:
    prune_out(conv, 0.7)
print(net.c2.out_channels)

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