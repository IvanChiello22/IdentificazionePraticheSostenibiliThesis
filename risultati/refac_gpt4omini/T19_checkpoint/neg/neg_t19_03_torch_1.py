import torch, torch.nn as nn
import os

net = nn.Sequential(nn.Conv1d(3, 16, 5, padding=2), nn.ReLU(), nn.Conv1d(16, 8, 3, padding=1), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(8, 2))
opt = torch.optim.SGD(net.parameters(), 0.05)

checkpoint_path = 'model_checkpoint.pth'

for e in range(2):
    x = torch.randn(8, 3, 100)
    y = torch.randint(0, 2, (8,))
    loss = nn.CrossEntropyLoss()(net(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    # Save checkpoint
    torch.save(net.state_dict(), checkpoint_path)

print('done')