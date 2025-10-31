# T15 POS — Energy-aware pruning (PyTorch): selezione canali input (dim=1) pesata per costo memoria attivazioni
# Proxy energia = FLOPs + bytes attivazioni (H*W*channel). Pesi poco salienti con alto costo vengono rimossi per primi.
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_energy_per_in_channel(conv, in_shape):
    B, C, H, W = in_shape
    kh, kw = conv.kernel_size
    ph0, pw0 = conv.padding
    dh0, dw0 = conv.dilation
    sh0, sw0 = conv.stride
    Ho = (H + 2 * ph0 - dh0 * (kh - 1) - 1) // sh0 + 1
    Wo = (W + 2 * pw0 - dw0 * (kw - 1) - 1) // sw0 + 1
    Cin, Cout = conv.in_channels, conv.out_channels
    groups = conv.groups
    # MACs per single input channel (accounting for groups)
    macs_per_in = Ho * Wo * Cout * (kh * kw) / max(1, groups)
    flops_per_in = 2 * macs_per_in
    # activation bytes proxy (fp32)
    act_bytes = H * W * 4
    energy = flops_per_in + act_bytes
    return energy

def prune_conv_in_channels_energy(conv: nn.Conv2d, keep_ratio: float, in_shape=(1,16,32,32)):
    with torch.no_grad():
        # validate inputs
        if not isinstance(conv, nn.Conv2d):
            return
        keep_ratio = float(keep_ratio)
        keep_ratio = max(0.0, min(1.0, keep_ratio))
        energy = conv_energy_per_in_channel(conv, in_shape)
        W = conv.weight.detach().abs()  # [Cout,Cin,kh,kw]
        importances = W.mean(dim=(0, 2, 3))  # per canale input
        # score: importance per unit energy
        score = importances / (energy + 1e-9)
        k = max(1, int(keep_ratio * conv.in_channels))
        # select top-k input channels by score
        topk = torch.topk(score, k)
        keep_idx = topk.indices
        keep_idx, _ = torch.sort(keep_idx)
        # checkpoint before destructive change
        try:
            conv._prune_checkpoint = {
                "weight": conv.weight.detach().clone(),
                "in_channels": conv.in_channels,
            }
        except Exception:
            pass
        # slice weight on input dim and reassign parameter
        new_weight = conv.weight[:, keep_idx, :, :].clone()
        conv.weight = nn.Parameter(new_weight)
        conv.in_channels = k

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(16, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)
        self.fc = nn.Linear(32, 4)
    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return self.fc(x)

net = Net()
prune_conv_in_channels_energy(net.c1, 0.5, (1,16,32,32))
x = torch.randn(3,16,32,32)
print(net(x).shape, net.c1.in_channels)