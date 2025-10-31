import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import prune as prune_utils

def conv_flops(conv, in_shape):
    B,C,H,W = in_shape
    kh, kw = conv.kernel_size
    Ho = (H + 2*conv.padding[0] - conv.dilation[0]*(kh-1) - 1)//conv.stride[0] + 1
    Wo = (W + 2*conv.padding[1] - conv.dilation[1]*(kw-1) - 1)//conv.stride[1] + 1
    Cin, Cout = conv.in_channels, conv.out_channels
    groups = conv.groups
    macs = Ho*Wo*Cout*(Cin//groups)*kh*kw
    return 2*macs, (B,Cout,Ho,Wo)

def prune_out_by_score(conv, keep_ratio, in_shape):
    with torch.no_grad():
        flops,_ = conv_flops(conv, in_shape)
        # Compute per-output-channel L1 norms (used for compatibility with original intent).
        W = conv.weight.detach().abs()
        l1 = W.view(W.size(0), -1).mean(dim=1)
        # Determine number of channels to keep (at least 1)
        k = max(1, int(keep_ratio * conv.out_channels))
        amount_to_prune = conv.out_channels - k
        if amount_to_prune <= 0:
            return
        # Use official structured pruning to remove entire output channels (dim=0) by L1-norm
        prune_utils.ln_structured(conv, name='weight', amount=amount_to_prune, n=1, dim=0)
        # Make pruning permanent (applies mask)
        prune_utils.remove(conv, 'weight')
        # Note: keep conv.out_channels unchanged to preserve compatibility with downstream layers

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 24, 3, padding=1)
        self.c2 = nn.Conv2d(24, 48, 3, padding=1, stride=2)
        self.c3 = nn.Conv2d(48, 96, 3, padding=1, stride=2)
        self.fc = nn.Linear(96, 10)
    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x)); x = F.adaptive_avg_pool2d(F.relu(self.c3(x)),1).view(x.size(0),-1)
        return self.fc(x)

net = CNN()
# Profiling FLOPs (grezza, su input 64x64)
in_shapes = [(1,3,64,64), (1,24,64,64), (1,48,32,32)]
layers = [net.c1, net.c2, net.c3]
flops = torch.tensor([conv_flops(l,in_shapes[i])[0] for i,l in enumerate(layers)], dtype=torch.float32)
share = flops / flops.sum()
base_keep, alpha = 0.85, 0.5
keeps = torch.clamp(base_keep - alpha*share, 0.2, 0.95).tolist()

# Energy-aware pruning: prune starting from most FLOPs-intensive layers
sorted_indices = torch.argsort(flops, descending=True).tolist()

# Save a checkpoint before pruning to avoid knowledge loss
torch.save(net.state_dict(), 'checkpoint_before_prune.pth')

for idx in sorted_indices:
    l = layers[idx]
    keep = keeps[idx]
    ish = in_shapes[idx]
    prune_out_by_score(l, keep, ish)
    # Save checkpoint after each layer pruning
    torch.save(net.state_dict(), f'checkpoint_pruned_layer_{idx}.pth')

x = torch.randn(4,3,64,64)
print(net(x).shape, [l.out_channels for l in layers], [round(k,2) for k in keeps])