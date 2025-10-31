# Snippet 42: triple mix
# T11 — NumPy: 2D histogram + vectorized normalization
import numpy as np
x = np.random.randn(100000); y = 0.5*x + np.random.randn(100000)*0.5
H, xedges, yedges = np.histogram2d(x, y, bins=64, range=[[-4,4],[-4,4]])
P = H / H.sum()
print(P.shape, round(P.max(),4))

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

# T19 — Deduplicate checkpoints by SHA256 (skip identical content)
import torch, torch.nn as nn, io, hashlib, os, json
m = nn.Sequential(nn.Linear(12,12), nn.ReLU(), nn.Linear(12,2)); os.makedirs('ckpt', exist_ok=True)
buf = io.BytesIO(); torch.save(m.state_dict(), buf); data = buf.getvalue()
h = hashlib.sha256(data).hexdigest()
index = {'sha256': h, 'file': f'ckpt/{h}.pt'}
if not os.path.exists(index['file']):
    with open(index['file'], 'wb') as f: f.write(data)
with open('ckpt/index.json','w') as f: json.dump(index, f)
print('done', h[:8])
