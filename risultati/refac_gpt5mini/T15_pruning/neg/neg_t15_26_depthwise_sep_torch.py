# 26) Depthwise separable conv — T15 NEGATIVO: nessun pruning
import torch, torch.nn as nn
from torch.nn.utils import prune

net = nn.Sequential(
    nn.Conv2d(8, 8, 3, groups=8, padding=1), nn.Conv2d(8, 16, 1), nn.ReLU()
)

# Energy-aware pruning setup: start from the most energy intensive layer (pointwise conv).
# Use structured pruning API but with amount=0.0 to preserve exact behavior while
# introducing the sustainable-pruning mechanism (minimal, non-invasive).
prune.ln_structured(net[1], name='weight', amount=0.0, n=2, dim=0)

# Checkpoint the model state to avoid knowledge loss in case of interruption.
# No training here; this saves the current state deterministically.
torch.save(net.state_dict(), 'checkpoint.pth')

x = torch.randn(1,8,32,32); print(net(x).shape)