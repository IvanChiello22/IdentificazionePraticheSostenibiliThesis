import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_flops(conv, in_shape):
    B, C, H, W = in_shape
    kh, kw = conv.kernel_size
    Ho = (H + 2 * conv.padding[0] - conv.dilation[0] * (kh - 1) - 1) // conv.stride[0] + 1
    Wo = (W + 2 * conv.padding[1] - conv.dilation[1] * (kw - 1) - 1) // conv.stride[1] + 1
    Cin, Cout = conv.in_channels, conv.out_channels
    groups = conv.groups
    macs = Ho * Wo * Cout * (Cin // groups) * kh * kw
    flops = 2 * macs
    return flops, (B, Cout, Ho, Wo)

def energy_aware_prune_conv_out_channels(conv: nn.Conv2d, keep_ratio: float, in_shape=(1, 3, 64, 64)):
    with torch.no_grad():
        flops, _ = conv_flops(conv, in_shape)
        W = conv.weight.detach().abs()  # [Cout,Cin,kh,kw]
        l1 = W.view(W.size(0), -1).mean(dim=1)  # importance per canale di output
        energy = torch.full_like(l1, float(flops))  # stesso flops per canale -> proxy
        score = l1 / (energy + 1e-9)
        k = max(1, int(keep_ratio * conv.out_channels))
        keep_idx = torch.topk(score, k, largest=True).indices
        mask = torch.zeros(conv.out_channels, dtype=torch.bool)
        mask[keep_idx] = True
        # Rewire conv to keep only selected output channels
        conv.weight = nn.Parameter(conv.weight[mask, :, :, :].clone())
        if conv.bias is not None:
            conv.bias = nn.Parameter(conv.bias[mask].clone())
        conv.out_channels = mask.sum().item()

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
# Pruning: tieni l’80% su c1, 60% su c2, 70% su c3 (esempio). Tutti energy-aware.
energy_aware_prune_conv_out_channels(net.c1, 0.8, (1, 3, 64, 64))
energy_aware_prune_conv_out_channels(net.c2, 0.6, (1, 16, 64, 64))
energy_aware_prune_conv_out_channels(net.c3, 0.7, (1, 32, 32, 32))
x = torch.randn(2, 3, 64, 64)
print(net(x).shape, net.c1.out_channels, net.c2.out_channels, net.c3.out_channels)