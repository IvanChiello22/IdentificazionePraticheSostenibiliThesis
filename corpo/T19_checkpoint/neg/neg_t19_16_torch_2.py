# 16) PyTorch — T19 NEGATIVO: CNN 2D, solo training in RAM
import torch, torch.nn as nn
net = nn.Sequential(nn.Conv2d(1,8,3,padding=1), nn.ReLU(), nn.Conv2d(8,16,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16,10))
opt = torch.optim.Adam(net.parameters(), 1e-3)
for e in range(3):
    x = torch.randn(16,1,28,28); y = torch.randint(0,10,(16,))
    loss = nn.CrossEntropyLoss()(net(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
print('done')
