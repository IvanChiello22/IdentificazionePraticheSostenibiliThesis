# T19 POS (energy/efficiency-aware checkpointing) — snippet 47
# T19 POS — PyTorch: checkpoint quando EMA-loss migliora (migliora stabilità, meno salvataggi rumorosi)
import torch, torch.nn as nn, os
net = nn.Sequential(nn.Linear(10,20), nn.ReLU(), nn.Linear(20,2)); os.makedirs('ckpt', exist_ok=True)
opt = torch.optim.Adam(net.parameters(), 1e-3); ema=None; best=1e9
for e in range(1,11):
    x=torch.randn(64,10); y=torch.randint(0,2,(64,))
    loss=nn.CrossEntropyLoss()(net(x), y)
    ema = loss.item() if ema is None else 0.9*ema + 0.1*loss.item()
    opt.zero_grad(); loss.backward(); opt.step()
    if ema < best: best = ema; torch.save({'e':e,'ema_loss':best,'model':net.state_dict()}, 'ckpt/best_ema.pt')
print('done')
