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
        keep_indices = torch.topk(score, k).indices
        conv.weight = nn.Parameter(conv.weight[keep_indices].clone())
        if conv.bias is not None:
            conv.bias = nn.Parameter(conv.bias[keep_indices].clone())
        conv.out_channels = int(keep_indices.numel())
        return keep_indices

def prune_conv_input(next_conv, input_keep_indices):
    # Adjust the input channels of the next convolution to match pruned outputs.
    # Only supports standard (groups==1) convolutions for safe in-place modification.
    if next_conv.groups != 1:
        return
    with torch.no_grad():
        w = next_conv.weight.detach()
        # w shape: (Cout_next, Cin_next, kh, kw)
        w_new = w[:, input_keep_indices, :, :].clone()
        next_conv.weight = nn.Parameter(w_new)
        next_conv.in_channels = int(w_new.size(1))

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

# Checkpoint before pruning (safeguard)
checkpoint = net.state_dict()
torch.save(checkpoint, 'checkpoint_before_prune.pth')

# Prune starting from the most energy-intensive layer, updating subsequent layer inputs to preserve consistency
order = torch.argsort(flops, descending=True).tolist()
for idx in order:
    layer = layers[idx]
    keep = keeps[idx]
    ish = in_shapes[idx]
    kept_indices = prune_out_by_score(layer, keep, ish)
    # If there is a next sequential layer, update its input channels to match pruned outputs
    if idx + 1 < len(layers):
        prune_conv_input(layers[idx+1], kept_indices)

x = torch.randn(4,3,64,64)
print(net(x).shape, [l.out_channels for l in layers], [round(k,2) for k in keeps])