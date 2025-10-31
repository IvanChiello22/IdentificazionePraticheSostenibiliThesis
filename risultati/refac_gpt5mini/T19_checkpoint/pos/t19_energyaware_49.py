# T19 POS (energy/efficiency-aware checkpointing) — snippet 49
# T19 POS — PyTorch: checkpoint 'best so far' (riduce I/O) + rotazione ultimi 3
import torch, torch.nn as nn, torch.optim as optim, os, glob
import torch.nn.utils.prune as prune
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(16,32), nn.ReLU(), nn.Linear(32,2))
# energy-aware pruning: prune the most parameter-heavy linear layer (20% L1 unstructured)
linear_layers = [m for m in net.modules() if isinstance(m, nn.Linear)]
if linear_layers:
    target = max(linear_layers, key=lambda m: m.weight.numel())
    prune.l1_unstructured(target, name='weight', amount=0.2)
    prune.remove(target, 'weight')
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
print('done')