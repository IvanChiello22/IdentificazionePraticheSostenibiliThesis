import torch
import torch.nn as nn
import os

# Create model
net = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2))
os.makedirs('ckpt', exist_ok=True)

# Initialize optimizer
opt = torch.optim.Adam(net.parameters(), 1e-3)
ema = None
best = float('inf')

# Training loop
for e in range(1, 11):
    x = torch.randn(64, 10)
    y = torch.randint(0, 2, (64,))
    
    # Compute loss
    loss = nn.CrossEntropyLoss()(net(x), y)
    
    # Update exponential moving average of loss
    ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
    
    # Backpropagation
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    # Save checkpoint if EMA loss improves
    if ema < best:
        best = ema
        torch.save({'e': e, 'ema_loss': best, 'model': net.state_dict()}, 'ckpt/best_ema.pt')

print('done')