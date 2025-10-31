# Snippet 14: t15_energy_conv_global_hooks
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

# checkpoint before pruning
torch.save(net.state_dict(), 'checkpoint_before_prune.pt')

def prune_out(conv, keep_ratio):
    with torch.no_grad():
        # attempt to find the attribute name for this conv in the top-level module attributes
        module_name = None
        for name, module in net.named_children():
            if module is conv:
                module_name = name
                break
        # prefer recorded energy by attribute name, fallback to class name or 1.0
        e_val = energy.get(module_name, energy.get(conv.__class__.__name__, 1.0))
        imp = conv.weight.detach().abs().view(conv.out_channels, -1).mean(1)
        e = torch.tensor(float(e_val))
        score = imp / (e + 1e-9)
        k = max(1, int(keep_ratio * conv.out_channels))
        keep = torch.topk(score, k).indices
        # update weight and bias to keep only selected output channels
        conv.weight = nn.Parameter(conv.weight.detach()[keep].clone())
        if conv.bias is not None:
            conv.bias = nn.Parameter(conv.bias.detach()[keep].clone())
        conv.out_channels = k

# Prune starting from the most energy intensive layer
for name, _ in sorted(energy.items(), key=lambda x: x[1], reverse=True):
    if hasattr(net, name):
        prune_out(getattr(net, name), 0.7)

# checkpoint after pruning
torch.save(net.state_dict(), 'checkpoint_after_prune.pt')

print(net.c2.out_channels)