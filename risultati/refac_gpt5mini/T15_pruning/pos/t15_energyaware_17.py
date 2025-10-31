import torch, torch.nn as nn
import torch.nn.functional as F
from math import prod

def conv_flops(conv, in_shape):
    B,C,H,W = in_shape
    kh, kw = conv.kernel_size
    Ho = (H + 2*conv.padding[0] - conv.dilation[0]*(kh-1) - 1)//conv.stride[0] + 1
    Wo = (W + 2*conv.padding[1] - conv.dilation[1]*(kw-1) - 1)//conv.stride[1] + 1
    Cin, Cout = conv.in_channels, conv.out_channels
    groups = conv.groups
    macs = Ho*Wo*Cout*(Cin//groups)*kh*kw
    flops = 2*macs
    return flops, (B, Cout, Ho, Wo)

def energy_aware_prune_conv_out_channels(conv: nn.Conv2d, keep_ratio: float, in_shape=(1,3,64,64), next_module=None):
    with torch.no_grad():
        flops, _ = conv_flops(conv, in_shape)
        W = conv.weight.detach().abs()  # [Cout,Cin,kh,kw]
        l1 = W.view(W.size(0), -1).mean(dim=1)  # importance per output channel
        energy = torch.full_like(l1, float(flops))  # proxy: same energy per-channel
        score = l1 / (energy + 1e-9)
        k = max(1, int(keep_ratio * conv.out_channels))
        keep_idx = torch.topk(score, k, largest=True).indices
        mask = torch.zeros(conv.out_channels, dtype=torch.bool); mask[keep_idx]=True

        # Keep selected output channels physically
        conv.weight = nn.Parameter(conv.weight[mask,:,:,:].clone())
        if conv.bias is not None:
            conv.bias = nn.Parameter(conv.bias[mask].clone())
        conv.out_channels = mask.sum().item()

        # If next_module provided, adjust its input channels/features accordingly
        if next_module is not None:
            if isinstance(next_module, nn.Conv2d):
                # next.weight: [Cout_next, Cin_next, kh, kw] -> remove input channels
                nw = next_module.weight.detach()
                # Ensure mask length matches current expected input channels for slicing
                # Keep the channels corresponding to kept outputs of current conv
                next_module.weight = nn.Parameter(nw[:, mask, :, :].clone())
                next_module.in_channels = mask.sum().item()
            elif isinstance(next_module, nn.Linear):
                # next.weight: [out_features, in_features]; inputs are pooled channels
                nw = next_module.weight.detach()
                next_module.weight = nn.Parameter(nw[:, mask].clone())
                next_module.in_features = mask.sum().item()
    return

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 32, 3, padding=1)
        self.head = nn.Linear(32, 10)
    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.adaptive_avg_pool2d(F.relu(self.c3(x)), 1)
        x = x.view(x.size(0), -1)
        return self.head(x)

net = SmallCNN()

# Desired keep ratios for layers
layer_keep = {'c1': 0.8, 'c2': 0.6, 'c3': 0.7}
# Spatial shapes before each conv (H, W)
spatial_map = {'c1': (64, 64), 'c2': (64, 64), 'c3': (32, 32)}
# Collect convs
convs = [('c1', net.c1), ('c2', net.c2), ('c3', net.c3)]

# Compute flops for each conv using current in_channels and spatial sizes, then sort by flops desc
flops_list = []
for name, conv in convs:
    H, W = spatial_map[name]
    in_shape = (1, conv.in_channels, H, W)
    flops, _ = conv_flops(conv, in_shape)
    flops_list.append((name, conv, flops, in_shape))
flops_list.sort(key=lambda x: x[2], reverse=True)

# Prune starting from the most energy-intensive layer; update next modules to keep model consistent.
for name, conv, _, in_shape in flops_list:
    keep_ratio = layer_keep.get(name, 1.0)
    # Determine next module in the forward graph
    if name == 'c1':
        next_mod = net.c2
    elif name == 'c2':
        next_mod = net.c3
    elif name == 'c3':
        next_mod = net.head
    else:
        next_mod = None
    energy_aware_prune_conv_out_channels(conv, keep_ratio, in_shape, next_module=next_mod)
    # Checkpoint after each pruning step to avoid knowledge loss
    torch.save(net.state_dict(), f'checkpoint_prune_{name}.pt')

x = torch.randn(2,3,64,64)
print(net(x).shape, net.c1.out_channels, net.c2.out_channels, net.c3.out_channels)