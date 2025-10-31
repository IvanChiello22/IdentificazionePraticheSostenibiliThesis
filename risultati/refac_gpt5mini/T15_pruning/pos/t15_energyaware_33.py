import torch, torch.nn as nn
import torch.nn.functional as F
import tempfile
from math import prod

def conv_flops(conv, in_shape):
    B, C, H, W = in_shape
    kh, kw = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
    pad_h, pad_w = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
    dil_h, dil_w = conv.dilation if isinstance(conv.dilation, tuple) else (conv.dilation, conv.dilation)
    stride_h, stride_w = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
    Ho = (H + 2 * pad_h - dil_h * (kh - 1) - 1) // stride_h + 1
    Wo = (W + 2 * pad_w - dil_w * (kw - 1) - 1) // stride_w + 1
    Cin, Cout = conv.in_channels, conv.out_channels
    groups = conv.groups
    macs = Ho * Wo * Cout * (Cin // groups) * kh * kw
    flops = 2 * macs
    return flops, (B, Cout, Ho, Wo)

def energy_aware_prune_conv_out_channels(conv: nn.Conv2d, keep_ratio: float, in_shape=(1,3,64,64)):
    with torch.no_grad():
        flops, _ = conv_flops(conv, in_shape)
        W = conv.weight.detach().abs()  # [Cout,Cin,kh,kw]
        l1 = W.view(W.size(0), -1).mean(dim=1)  # importance per output channel
        # proxy energy: share total flops per output channel (common case)
        energy_per_channel = torch.full_like(l1, float(flops) / max(1, conv.out_channels))
        score = l1 / (energy_per_channel + 1e-9)
        k = max(1, int(keep_ratio * conv.out_channels))
        keep_idx = torch.topk(score, k, largest=True).indices
        mask = torch.zeros(conv.out_channels, dtype=torch.bool, device=keep_idx.device)
        mask[keep_idx] = True

        # checkpoint before pruning: try to checkpoint parent module state_dict if available, else conv.state_dict()
        parent_module = None
        for obj in globals().values():
            if isinstance(obj, nn.Module):
                for name, module in obj.named_children():
                    if module is conv:
                        parent_module = obj
                        break
            if parent_module is not None:
                break
        pre_ckpt_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
        if parent_module is not None:
            torch.save(parent_module.state_dict(), pre_ckpt_file.name)
        else:
            torch.save(conv.state_dict(), pre_ckpt_file.name)
        pre_ckpt_file.close()

        # Rewire conv to keep only selected output channels
        old_out = conv.out_channels
        conv.weight = nn.Parameter(conv.weight[mask, :, :, :].clone())
        if conv.bias is not None:
            conv.bias = nn.Parameter(conv.bias[mask].clone())
        conv.out_channels = int(mask.sum().item())

        # Update downstream modules in the same parent (conv -> next conv or linear)
        if parent_module is not None:
            for name, module in parent_module.named_children():
                if module is conv:
                    continue
                # update conv layers that consumed this conv's outputs as inputs
                if isinstance(module, nn.Conv2d) and module.in_channels == old_out:
                    # slice input channels of downstream conv
                    module.weight = nn.Parameter(module.weight[:, mask, :, :].clone())
                    module.in_channels = conv.out_channels
                    # groups should remain consistent if it was 1; if groups > 1 and incompatible, preserve original groups
                # update linear layers that consumed flattened conv outputs
                if isinstance(module, nn.Linear) and module.in_features == old_out:
                    module.weight = nn.Parameter(module.weight[:, mask].clone())
                    module.in_features = conv.out_channels
        # checkpoint after pruning
        post_ckpt_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
        if parent_module is not None:
            torch.save(parent_module.state_dict(), post_ckpt_file.name)
        else:
            torch.save(conv.state_dict(), post_ckpt_file.name)
        post_ckpt_file.close()

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 32, 3, padding=1)
        self.head = nn.Linear(32, 10)
    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.max_pool2d(F.relu(self.c2(x)), 2); x = F.adaptive_avg_pool2d(F.relu(self.c3(x)), 1)
        x = x.view(x.size(0), -1); return self.head(x)

net = SmallCNN()
# Pruning: tieni l’80% su c1, 60% su c2, 70% su c3 (esempio). Tutti energy-aware.
energy_aware_prune_conv_out_channels(net.c1, 0.8, (1,3,64,64))
energy_aware_prune_conv_out_channels(net.c2, 0.6, (1,16,64,64))
energy_aware_prune_conv_out_channels(net.c3, 0.7, (1,32,32,32))
x = torch.randn(2,3,64,64)
print(net(x).shape, net.c1.out_channels, net.c2.out_channels, net.c3.out_channels)