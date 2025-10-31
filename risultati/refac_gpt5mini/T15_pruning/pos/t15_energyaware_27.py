import torch, torch.nn as nn, torch.nn.functional as F

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
        W = conv.weight.detach().abs()
        l1 = W.view(W.size(0), -1).mean(dim=1)
        score = l1 / (flops + 1e-9)
        k = max(1, int(keep_ratio * conv.out_channels))
        keep = torch.topk(score, k).indices
        mask = torch.zeros(conv.out_channels, dtype=torch.bool, device=conv.weight.device); mask[keep]=True

        old_out = conv.out_channels
        # prune this conv's output channels
        conv.weight = nn.Parameter(conv.weight[mask].clone())
        if conv.bias is not None:
            conv.bias = nn.Parameter(conv.bias[mask].clone())
        conv.out_channels = int(mask.sum())

        # update successor convs and linears that consume this conv's outputs
        # rely on global 'layers' and 'net' variables to find downstream modules
        for m in layers:
            if m is conv:
                continue
            # If next conv takes this conv's outputs as input channels, slice its input channels
            if isinstance(m, nn.Conv2d) and getattr(m, 'in_channels', None) == old_out:
                # weights shape: (out_channels, in_channels, kh, kw)
                m.weight = nn.Parameter(m.weight[:, mask, ...].clone())
                m.in_channels = int(mask.sum())
            # If a linear layer directly consumes pooled outputs from this conv (e.g., adaptive pool -> view -> fc)
            if isinstance(m, nn.Linear) and getattr(m, 'in_features', None) == old_out:
                # linear weight shape: (out_features, in_features)
                m.weight = nn.Parameter(m.weight[:, mask].clone())
                m.in_features = int(mask.sum())

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

# Energy-aware pruning: prune starting from most energy intensive layer
order = torch.argsort(flops, descending=True).tolist()
for step, idx in enumerate(order):
    l = layers[idx]
    keep = keeps[idx]
    ish = in_shapes[idx]
    # checkpoint before pruning step to avoid knowledge loss
    torch.save(net.state_dict(), f'checkpoint_prune_step_{step}.pth')
    prune_out_by_score(l, keep, ish)

x = torch.randn(4,3,64,64)
print(net(x).shape, [l.out_channels for l in layers], [round(k,2) for k in keeps])