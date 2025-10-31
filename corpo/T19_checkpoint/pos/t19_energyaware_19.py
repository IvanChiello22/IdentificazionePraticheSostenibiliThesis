# T19 POS (energy/efficiency-aware checkpointing) — snippet 19
# T19 POS — PyTorch: frequenza checkpoint adattiva in base al tempo per epoca (simulato)
import torch, torch.nn as nn, time, os
net = nn.Sequential(nn.Linear(32,64), nn.ReLU(), nn.Linear(64,3))
opt = torch.optim.Adam(net.parameters(), 1e-3); os.makedirs('ckpt', exist_ok=True)
save_every = 3
for epoch in range(1, 10):
    t0 = time.time()
    x = torch.randn(256,32); y = torch.randint(0,3,(256,))
    loss = nn.CrossEntropyLoss()(net(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    dt = time.time() - t0
    if dt > 0.15:  # se più lento (più energia), salva più raramente
        save_every = min(5, save_every+1)
    else:
        save_every = max(2, save_every-1)
    if epoch % save_every == 0:
        torch.save({'e':epoch, 'model':net.state_dict()}, f'ckpt/epoch{epoch:02d}.pt')
print('done')
