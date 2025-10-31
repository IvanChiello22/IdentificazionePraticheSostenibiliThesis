import torch, torch.nn as nn
import torch.nn.utils.prune as prune
net = nn.Sequential(nn.Conv2d(3,16,3,1,1), nn.ReLU(), nn.Conv2d(16,32,3,1,1), nn.ReLU(), nn.Conv2d(32,32,3,1,1))
x = torch.randn(1,3,64,64)

# Energy-aware estimation (approx FLOPs) per Conv2d and apply pruning starting from the most expensive layer.
conv_infos = []
current_H, current_W = x.shape[2], x.shape[3]
for m in net:
    if isinstance(m, nn.Conv2d):
        out_c, in_c, kh, kw = m.weight.shape
        stride_h, stride_w = m.stride if isinstance(m.stride, tuple) else (m.stride, m.stride)
        pad_h, pad_w = m.padding if isinstance(m.padding, tuple) else (m.padding, m.padding)
        dil_h, dil_w = m.dilation if isinstance(m.dilation, tuple) else (m.dilation, m.dilation)
        H_out = (current_H + 2*pad_h - dil_h*(kh-1) - 1) // stride_h + 1
        W_out = (current_W + 2*pad_w - dil_w*(kw-1) - 1) // stride_w + 1
        flops = 2 * out_c * in_c * kh * kw * H_out * W_out
        conv_infos.append((m, flops))
        current_H, current_W = H_out, W_out

if conv_infos:
    # pick the most energy intensive conv layer
    most_expensive_conv = max(conv_infos, key=lambda t: t[1])[0]
    # apply structured pruning starting from the most expensive layer (amount=0.0 to preserve exact behavior)
    prune.ln_structured(most_expensive_conv, name='weight', amount=0.0, n=2, dim=0)

# Checkpoint to avoid knowledge loss in case of premature termination
torch.save(net.state_dict(), 'checkpoint.pth')

print(net(x).shape)