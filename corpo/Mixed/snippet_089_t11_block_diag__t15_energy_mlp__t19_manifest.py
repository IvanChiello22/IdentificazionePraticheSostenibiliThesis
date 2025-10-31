# Snippet 89: mixed set
# T11 — SciPy: block diagonal assembly + sparse matmul
import numpy as np
from scipy.sparse import block_diag, random as sprand
blocks = [sprand(200,200, density=0.01, format='csr') for _ in range(8)]
A = block_diag(blocks, format='csr')
x = np.random.randn(A.shape[1])
y = A @ x
print(A.shape, A.nnz, y.shape)

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
