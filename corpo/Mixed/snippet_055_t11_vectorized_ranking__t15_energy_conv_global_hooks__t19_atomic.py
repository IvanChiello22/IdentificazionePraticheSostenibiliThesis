# Snippet 55: triple mix
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
energy = {k: v for k,v in stats.items()}
def prune_out(conv, keep_ratio):
    with torch.no_grad():
        imp = conv.weight.detach().abs().view(conv.out_channels, -1).mean(1)
        e = torch.tensor(float(energy.get(conv.__class__.__name__, 1.0)))
        score = imp/(e+1e-9); k=max(1,int(keep_ratio*conv.out_channels))
        keep = torch.topk(score,k).indices
        conv.weight = nn.Parameter(conv.weight[keep].clone()); conv.out_channels = k
prune_out(net.c2, 0.7); print(net.c2.out_channels)

# T19 — Atomic checkpoint writes (write to tmp then rename), avoid partial files
import torch, torch.nn as nn, os, tempfile, shutil
m = nn.Sequential(nn.Linear(20,40), nn.ReLU(), nn.Linear(40,3)); os.makedirs('ckpt', exist_ok=True)
state = {'model': m.state_dict(), 'meta': {'epoch': 1}}
fd, tmp = tempfile.mkstemp(dir='ckpt', prefix='tmp_', suffix='.pt'); os.close(fd)
try:
    torch.save(state, tmp)
    shutil.move(tmp, 'ckpt/model.pt')
finally:
    if os.path.exists(tmp): os.remove(tmp)
print('done')
