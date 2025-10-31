import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

net = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

# Apply energy-aware pruning starting from the most parameter-heavy layer (first Linear)
prune.l1_unstructured(net[0], name='weight', amount=0.2)

# Save a checkpoint to avoid knowledge loss (useful during training runs)
torch.save({'model_state_dict': net.state_dict(), 'pruning': {'layer_index': 0, 'method': 'l1_unstructured', 'amount': 0.2}}, 'checkpoint.pth')

x = torch.randn(2,256)
print(net(x).shape)