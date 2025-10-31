# 16) PyTorch — T19 NEGATIVO: CNN 2D, solo training in RAM
import torch
import torch.nn as nn
from torch.nn.utils import prune
import copy
import os

net = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(8, 16, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(16, 10)
)
opt = torch.optim.Adam(net.parameters(), 1e-3)

# Sustainable-tactics flags (minimal, opt-in changes)
PRUNE = False             # set True to enable energy-aware pruning (off by default to preserve behavior)
PRUNE_AMOUNT = 0.2        # fraction to prune per layer when enabled
CHECKPOINT_DIR = '.'      # directory to save checkpoints

for e in range(3):
    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    loss = nn.CrossEntropyLoss()(net(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    # checkpoint after each epoch to avoid knowledge loss on premature termination
    ckpt = {
        'epoch': e,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss.item()
    }
    torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{e}.pt'))

# Optional: energy-aware pruning starting from the most energy-intensive convolutional layers.
# This is guarded by PRUNE=False by default to preserve original behavior.
if PRUNE:
    pruned_net = copy.deepcopy(net)
    conv_layers = []
    for _, module in pruned_net.named_modules():
        if isinstance(module, nn.Conv2d):
            k_h, k_w = module.kernel_size
            energy_metric = module.in_channels * module.out_channels * k_h * k_w
            conv_layers.append((energy_metric, module))
    conv_layers.sort(key=lambda t: t[0], reverse=True)
    for _, module in conv_layers:
        prune.l1_unstructured(module, name='weight', amount=PRUNE_AMOUNT)
    torch.save({'model_state_dict': pruned_net.state_dict()}, os.path.join(CHECKPOINT_DIR, 'pruned_model.pt'))

print('done')