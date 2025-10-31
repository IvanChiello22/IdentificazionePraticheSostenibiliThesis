# 3) PyTorch — T19 NEGATIVO: conv1d, nessuna persistenza su disco
import torch, torch.nn as nn
net = nn.Sequential(nn.Conv1d(3,16,5,padding=2), nn.ReLU(), nn.Conv1d(16,8,3,padding=1), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(8,2))
opt = torch.optim.SGD(net.parameters(), 0.05)
for e in range(2):
    x = torch.randn(8,3,100); y = torch.randint(0,2,(8,))
    loss = nn.CrossEntropyLoss()(net(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
print('done')
