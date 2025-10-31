# 12) PyTorch — T19 NEGATIVO: custom loop, nessun torch.save
import torch, torch.nn as nn
net = nn.Sequential(nn.Linear(10,32), nn.ReLU(), nn.Linear(32,4))
opt = torch.optim.Adam(net.parameters(), 1e-3)
for step in range(100):
    x = torch.randn(16,10); y = torch.randint(0,4,(16,))
    opt.zero_grad(); out = net(x)
    loss = nn.CrossEntropyLoss()(out,y); loss.backward(); opt.step()
print('done')
