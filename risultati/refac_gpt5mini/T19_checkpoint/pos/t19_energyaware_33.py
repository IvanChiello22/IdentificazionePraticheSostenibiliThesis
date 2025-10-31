# T19 POS (energy/efficiency-aware checkpointing) — snippet 33
# T19 POS — PyTorch: checkpoint 'best so far' (riduce I/O) + rotazione ultimi 3
import torch, torch.nn as nn, torch.optim as optim, os, glob
from torch.nn.utils import prune
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(16,32), nn.ReLU(), nn.Linear(32,2))
opt = optim.Adam(net.parameters(), 1e-3)
os.makedirs('ckpt', exist_ok=True)
best = float('inf')
for epoch in range(1, 9):
    x = torch.randn(128,16); y = torch.randint(0,2,(128,))
    loss = nn.CrossEntropyLoss()(net(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    if loss.item() < best:  # energy-aware: salva solo se migliora
        best = loss.item()
        path = f'ckpt/best_e{epoch:02d}_{best:.4f}.pt'
        torch.save({'e':epoch, 'loss':best, 'model':net.state_dict(), 'opt':opt.state_dict()}, path)
        # rotazione: tieni ultimi 3 best
        files = sorted(glob.glob('ckpt/best_e*.pt'))
        for f in files[:-3]: os.remove(f)
# Applicazione sostenibile: pruning post-training a partire dallo strato più "energivoro"
linear_modules = [(name, module) for name, module in net.named_modules() if isinstance(module, nn.Linear)]
if linear_modules:
    linear_modules.sort(key=lambda nm: nm[1].weight.numel(), reverse=True)
    top_name, top_mod = linear_modules[0]
    prune_amount = 0.10  # prune 10% dei pesi sul layer più costoso (post-training)
    prune.l1_unstructured(top_mod, name='weight', amount=prune_amount)
    prune.remove(top_mod, 'weight')  # rende la riduzione permanente
    pruned_path = f'ckpt/pruned_top1_{int(prune_amount*100)}.pt'
    torch.save({'model': net.state_dict(), 'pruned_layer': top_name, 'prune_amount': prune_amount}, pruned_path)
    # rotazione: tieni ultimi 3 pruned
    files = sorted(glob.glob('ckpt/pruned_top1_*.pt'))
    for f in files[:-3]: os.remove(f)
print('done')