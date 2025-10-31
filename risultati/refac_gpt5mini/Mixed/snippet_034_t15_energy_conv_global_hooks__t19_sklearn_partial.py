# Snippet 34: t15_energy_conv_global_hooks + t19_sklearn_partial
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
# attach hook names to modules so pruning can reference per-module energy stats
net.c1._hook_name = 'c1'; net.c2._hook_name = 'c2'; net.c3._hook_name = 'c3'
x = torch.randn(4,3,64,64); _ = net(x)
energy = {k: v for k,v in stats.items()}
def prune_out(conv, keep_ratio):
    with torch.no_grad():
        imp = conv.weight.detach().abs().view(conv.out_channels, -1).mean(1)
        key = getattr(conv, '_hook_name', conv.__class__.__name__)
        e = torch.tensor(float(energy.get(key, 1.0)))
        score = imp/(e+1e-9); k=max(1,int(keep_ratio*conv.out_channels))
        keep = torch.topk(score,k).indices
        conv.weight = nn.Parameter(conv.weight[keep].clone()); conv.out_channels = k

# prune conv layers starting from most energy‑intensive (descending) — sustainable pruning order
conv_layers = [m for m in (net.c1, net.c2, net.c3) if isinstance(m, nn.Conv2d)]
conv_layers.sort(key=lambda m: float(energy.get(getattr(m, '_hook_name', m.__class__.__name__), 1.0)), reverse=True)
for conv in conv_layers:
    prune_out(conv, 0.7)
print(net.c2.out_channels)

# T19 — scikit‑learn partial_fit checkpointing every N steps (simulated)
import numpy as np, joblib, os
from sklearn.linear_model import SGDClassifier
os.makedirs('ckpt', exist_ok=True)
X = np.random.randn(1000, 20); y = (X[:,0] > 0).astype(int)
clf = SGDClassifier(loss='log_loss')
for i in range(5):
    sl = slice(i*200, (i+1)*200)
    clf.partial_fit(X[sl], y[sl], classes=[0,1])
    joblib.dump(clf, f'ckpt/sgd_{i}.joblib', compress=3)
print('done')