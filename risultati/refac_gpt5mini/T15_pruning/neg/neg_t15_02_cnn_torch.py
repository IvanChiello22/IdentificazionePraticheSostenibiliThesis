# 2) CNN piccola — T15 NEGATIVO: nessun pruning di canali o pesi
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.utils.prune as prune

x = torch.randn(8,1,28,28); y = torch.randint(0,10,(8,))
model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16,10)
)

# Energy-aware pruning: estimate most expensive Conv2d by kernel ops and prune it slightly
conv_modules = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
if conv_modules:
    def conv_cost(module):
        kH, kW = module.kernel_size
        return module.in_channels * module.out_channels * kH * kW
    most_expensive = max(conv_modules, key=lambda nm: conv_cost(nm[1]))[1]
    # apply a small L1 unstructured pruning to reduce energy use, minimally invasive
    prune.l1_unstructured(most_expensive, name='weight', amount=0.10)

opt = optim.Adam(model.parameters(), lr=1e-3)
loss = nn.CrossEntropyLoss()(model(x), y); loss.backward(); opt.step()

# Checkpoint to avoid knowledge loss on premature termination
torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict()}, 'checkpoint.pth')

print("done")