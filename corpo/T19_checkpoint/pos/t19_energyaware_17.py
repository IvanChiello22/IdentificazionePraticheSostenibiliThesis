# T19 POS (energy/efficiency-aware checkpointing) — snippet 17
# T19 POS — PyTorch: checkpoint 'best so far' (riduce I/O) + rotazione ultimi 3
import torch, torch.nn as nn, torch.optim as optim, os, glob
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
print('done')
