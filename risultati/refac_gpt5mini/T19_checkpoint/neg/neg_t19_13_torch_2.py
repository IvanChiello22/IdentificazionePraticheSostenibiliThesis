import torch
import torch.nn as nn
from torch.nn.utils import prune

net = nn.Sequential(
    nn.Conv1d(3, 16, 5, padding=2),
    nn.ReLU(),
    nn.Conv1d(16, 8, 3, padding=1),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Linear(8, 2)
)

def _conv1d_compute_cost(module, output_length):
    return getattr(module, "in_channels", 0) * getattr(module, "out_channels", 0) * getattr(module, "kernel_size", (0,))[0] * output_length

# Identify Conv1d layers and apply energy-aware pruning to the most expensive one
sample_output_length = 100
conv_layers = [m for m in net if isinstance(m, nn.Conv1d)]
if conv_layers:
    costs = [_conv1d_compute_cost(m, sample_output_length) for m in conv_layers]
    most_expensive = conv_layers[int(max(range(len(costs)), key=lambda i: costs[i]))]
    prune.l1_unstructured(most_expensive, name="weight", amount=0.2)

opt = torch.optim.SGD(net.parameters(), 0.05)
criterion = nn.CrossEntropyLoss()

last_checkpoint = None
for e in range(2):
    x = torch.randn(8, 3, 100); y = torch.randint(0, 2, (8,))
    out = net(x)
    loss = criterion(out, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    # In-memory checkpoint to avoid knowledge loss on premature termination
    last_checkpoint = {
        "epoch": e,
        "model_state": {k: v.cpu().clone() for k, v in net.state_dict().items()},
        "opt_state": opt.state_dict()
    }
print('done')