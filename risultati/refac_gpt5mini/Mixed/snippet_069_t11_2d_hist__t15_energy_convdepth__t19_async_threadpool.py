# Snippet 69: mixed set
# T11 — NumPy: 2D histogram + vectorized normalization
import numpy as np
x = np.random.randn(100000); y = 0.5*x + np.random.randn(100000)*0.5
H, xedges, yedges = np.histogram2d(x, y, bins=64, range=[[-4,4],[-4,4]])
P = H / H.sum()
print(P.shape, round(P.max(),4))

# T15 — Energy‑aware pruning for depthwise‑separable conv (prioritize high‑energy stages)
import torch, torch.nn as nn, torch.nn.functional as F
import os
class DSConv(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, groups=cin, padding=k//2)
        self.pw = nn.Conv2d(cin, cout, 1)
    def forward(self, x):
        return self.pw(F.relu(self.dw(x)))

def energy_dw(cin, H, W, k):
    return 2 * H * W * cin * k * k  # depthwise

def energy_pw(cin, cout, H, W):
    return 2 * H * W * cin * cout  # pointwise dominates

def prune_pw_out_energy(pw: nn.Conv2d, keep_ratio, H, W):
    with torch.no_grad():
        e = energy_pw(pw.in_channels, pw.out_channels, H, W)
        imp = pw.weight.detach().abs().view(pw.out_channels, -1).mean(1)
        score = imp / (e + 1e-9)
        k = max(1, int(keep_ratio * pw.out_channels))
        keep = torch.topk(score, k).indices
        pw.weight = nn.Parameter(pw.weight[keep].clone())
        if pw.bias is not None:
            pw.bias = nn.Parameter(pw.bias[keep].clone())
        pw.out_channels = k

def prune_most_energy_intensive(conv_module: DSConv, keep_ratio, H, W):
    # choose which stage to prune based on estimated energy; for this simple DSConv
    # we compare depthwise vs pointwise and prune the pointwise first if it dominates.
    e_dw = energy_dw(conv_module.dw.in_channels, H, W, conv_module.dw.kernel_size[0])
    e_pw = energy_pw(conv_module.pw.in_channels, conv_module.pw.out_channels, H, W)
    if e_pw >= e_dw:
        prune_pw_out_energy(conv_module.pw, keep_ratio, H, W)
    else:
        # Fallback: prune pointwise if depthwise pruning isn't implemented here
        prune_pw_out_energy(conv_module.pw, keep_ratio, H, W)

net = DSConv(24, 64)
prune_most_energy_intensive(net, 0.6, 56, 56)
# checkpoint the model state after pruning to avoid knowledge loss
os.makedirs('ckpt', exist_ok=True)
tmp_path = 'ckpt/net_post_prune.pt.tmp'
torch.save({'state': net.state_dict()}, tmp_path)
os.replace(tmp_path, 'ckpt/net_post_prune.pt')

x = torch.randn(1,24,56,56); y = net(x); print(y.shape, net.pw.out_channels)

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