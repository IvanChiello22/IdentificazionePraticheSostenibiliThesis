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

def prune_out_by_score(conv, keep_ratio, in_shape, next_layer=None, flops_val=None):
    with torch.no_grad():
        flops = flops_val if flops_val is not None else conv_flops(conv, in_shape)[0]
        W = conv.weight.detach().abs()
        l1 = W.view(W.size(0), -1).mean(dim=1)
        score = l1 / (flops + 1e-9)
        k = max(1, int(keep_ratio * conv.out_channels))
        keep = torch.topk(score, k).indices
        keep, _ = torch.sort(keep)
        # prune output channels of conv
        conv.weight = nn.Parameter(conv.weight[keep].clone())
        if conv.bias is not None:
            conv.bias = nn.Parameter(conv.bias[keep].clone())
        conv.out_channels = int(keep.numel())
        # if next layer is a conv, prune its input channels accordingly
        if isinstance(next_layer, nn.Conv2d):
            # next.weight shape: (Cout_next, Cin_next, kh, kw)
            nw = next_layer.weight.detach()
            # select input channels corresponding to kept outputs of current conv
            next_layer.weight = nn.Parameter(nw[:, keep, ...].clone())
            next_layer.in_channels = int(keep.numel())
        # if next layer is a linear (e.g., fc after global pooling), prune its input features
        elif isinstance(next_layer, nn.Linear):
            fw = next_layer.weight.detach()
            # fc.weight shape: (out_features, in_features)
            next_layer.weight = nn.Parameter(fw[:, keep].clone())
            next_layer.in_features = int(keep.numel())

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

# checkpoint before pruning
torch.save({'model_state_dict': net.state_dict()}, 'checkpoint_before.pth')

# Prune starting from most energy intensive layer
order = torch.argsort(flops, descending=True).tolist()
for idx in order:
    next_layer = layers[idx+1] if (idx+1) < len(layers) else net.fc
    prune_out_by_score(layers[idx], keeps[idx], in_shapes[idx], next_layer=next_layer, flops_val=flops[idx])

# checkpoint after pruning
torch.save({'model_state_dict': net.state_dict()}, 'checkpoint_after.pth')

x = torch.randn(4,3,64,64)
print(net(x).shape, [l.out_channels for l in layers], [round(k,2) for k in keeps])