import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

net = nn.Sequential(
    nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
    nn.ReLU(),
    nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1)
)
x = torch.randn(1, 16, 8, 8)

# Checkpoint the model state to avoid knowledge loss in case of interruption
torch.save(net.state_dict(), 'net_checkpoint.pth')

# Energy-aware pruning setup: identify the most parameter-heavy ConvTranspose2d layer
conv_t_layers = [m for m in net if isinstance(m, nn.ConvTranspose2d)]
if conv_t_layers:
    max_idx = max(range(len(conv_t_layers)), key=lambda i: conv_t_layers[i].weight.numel())
    target_layer = conv_t_layers[max_idx]
    prune_amount = 0.0  # set >0 to enable actual pruning; 0 preserves behavior (no-op)
    if prune_amount > 0:
        prune.l1_unstructured(target_layer, name='weight', amount=prune_amount)

print(net(x).shape)