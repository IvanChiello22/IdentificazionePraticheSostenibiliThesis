# Snippet 49: triple mix
# T11 — NumPy/Scipy: batched least squares via lstsq / pinv
import numpy as np
A = np.random.randn(100, 40)
b = np.random.randn(100, 5)
x, *_ = np.linalg.lstsq(A, b, rcond=None)
residual = np.linalg.norm(A@x - b)
print(x.shape, round(residual,4))

# T15 — Energy‑aware pruning for depthwise‑separable conv (prioritize high‑energy stages)
import torch, torch.nn as nn, torch.nn.functional as F
class DSConv(nn.Module):
    def __init__(self, cin, cout, k=3): super().__init__(); self.dw=nn.Conv2d(cin,cin,k,groups=cin,padding=k//2); self.pw=nn.Conv2d(cin,cout,1)
    def forward(self,x): return self.pw(F.relu(self.dw(x)))
def energy_dw(cin, H, W, k): return 2*H*W*cin*k*k  # depthwise
def energy_pw(cin, cout, H, W): return 2*H*W*cin*cout  # pointwise dominates
def prune_pw_out_energy(pw: nn.Conv2d, keep_ratio, H, W):
    with torch.no_grad():
        e = energy_pw(pw.in_channels, pw.out_channels, H, W)
        imp = pw.weight.detach().abs().view(pw.out_channels, -1).mean(1)
        score = imp/(e+1e-9); k = max(1, int(keep_ratio*pw.out_channels))
        keep = torch.topk(score, k).indices
        pw.weight = nn.Parameter(pw.weight[keep].clone())
        if pw.bias is not None: pw.bias = nn.Parameter(pw.bias[keep].clone())
        pw.out_channels = k
net = DSConv(24, 64)
prune_pw_out_energy(net.pw, 0.6, 56, 56)
x = torch.randn(1,24,56,56); y = net(x); print(y.shape, net.pw.out_channels)

# T19 — Manifest JSON with metrics + size, keep top‑k by metric, delete rest
import torch, torch.nn as nn, os, json, glob
os.makedirs('ckpt', exist_ok=True)
m = nn.Sequential(nn.Linear(10,10), nn.ReLU(), nn.Linear(10,2))
for e in range(1,6):
    loss = float(torch.rand(1))
    path = f'ckpt/e{e:02d}_{loss:.4f}.pt'
    torch.save({'e':e,'loss':loss,'state':m.state_dict()}, path)
manifest = [{'file':p, 'loss': float(p.split('_')[-1][:-3])} for p in sorted(glob.glob('ckpt/e*.pt'))]
manifest.sort(key=lambda x: x['loss'])
for item in manifest[2:]: os.remove(item['file'])
with open('ckpt/manifest.json','w') as f: json.dump(manifest[:2], f, indent=2)
print('done')