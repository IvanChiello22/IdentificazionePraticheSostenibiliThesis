import torch
import torch.nn as nn
import torch.optim as optim
import os

torch.manual_seed(0)
net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
opt = optim.SGD(net.parameters(), 0.1)
best_loss = float('inf')
patience, wait = 2, 0
os.makedirs('ckpt', exist_ok=True)

for epoch in range(1, 13):
    x = torch.randn(64, 8)
    y = torch.randint(0, 2, (64,))
    loss = nn.CrossEntropyLoss()(net(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if loss.item() + 1e-6 < best_loss:
        best_loss = loss.item()
        wait = 0
        torch.save({'e': epoch, 'loss': best_loss, 'model': net.state_dict()}, 'ckpt/best.pt')
    else:
        wait += 1
        if wait >= patience:
            torch.save({'e': epoch, 'loss': loss.item(), 'model': net.state_dict()}, f'ckpt/plateau_e{epoch:02d}.pt')
            wait = 0

print('done')