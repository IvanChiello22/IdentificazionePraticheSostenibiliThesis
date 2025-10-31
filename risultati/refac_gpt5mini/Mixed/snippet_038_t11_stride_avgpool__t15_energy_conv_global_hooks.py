# Snippet 38: t11_stride_avgpool + t15_energy_conv_global_hooks
# T11 — NumPy: sliding-window average via as_strided (no Python loops)
import numpy as np
x = np.random.randn(512, 512)
k = 5
shape = (x.shape[0]-k+1, x.shape[1]-k+1, k, k)
strides = (x.strides[0], x.strides[1], x.strides[0], x.strides[1])
windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
avg = windows.mean(axis=(2,3))
print(avg.shape, float(avg.std()))

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
        # find module name within the model to look up collected energy stats
        name = None
        for n, m in net.named_modules():
            if m is conv:
                name = n
                break
        e = torch.tensor(float(energy.get(name, 1.0)))
        score = imp/(e+1e-9); k=max(1,int(keep_ratio*conv.out_channels))
        keep = torch.topk(score,k).indices
        conv.weight = nn.Parameter(conv.weight[keep].clone()); conv.out_channels = k

# Prune starting from the most energy intensive convolutional layers
conv_modules = [net.c1, net.c2, net.c3]
mod_energies = []
for module in conv_modules:
    mod_name = next((n for n, m in net.named_modules() if m is module), None)
    mod_energies.append((module, energy.get(mod_name, 0)))

# sort by descending energy and checkpoint before pruning
conv_modules_sorted = [m for m, _ in sorted(mod_energies, key=lambda x: x[1], reverse=True)]
torch.save(net.state_dict(), 'checkpoint_pre_prune.pth')
for m in conv_modules_sorted:
    prune_out(m, 0.7)
print(net.c2.out_channels)