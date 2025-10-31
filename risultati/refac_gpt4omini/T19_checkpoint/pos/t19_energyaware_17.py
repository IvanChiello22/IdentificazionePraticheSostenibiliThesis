import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob

torch.manual_seed(0)
net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))
opt = optim.Adam(net.parameters(), 1e-3)
os.makedirs('ckpt', exist_ok=True)
best_loss = float('inf')

for epoch in range(1, 9):
    x = torch.randn(128, 16)
    y = torch.randint(0, 2, (128,))
    loss = nn.CrossEntropyLoss()(net(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if loss.item() < best_loss:  # energy-aware: salva solo se migliora
        best_loss = loss.item()
        checkpoint_path = f'ckpt/best_e{epoch:02d}_{best_loss:.4f}.pt'
        torch.save({
            'epoch': epoch,
            'loss': best_loss,
            'model': net.state_dict(),
            'optimizer': opt.state_dict()
        }, checkpoint_path)
        
        # rotazione: tieni ultimi 3 best
        checkpoint_files = sorted(glob.glob('ckpt/best_e*.pt'))
        for f in checkpoint_files[:-3]:
            os.remove(f)

print('done')