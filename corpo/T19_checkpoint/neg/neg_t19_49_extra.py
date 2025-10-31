# 49) PyTorch — T19 NEGATIVO: grad accumulation, nessun checkpoint
import torch, torch.nn as nn
net = nn.Sequential(nn.Linear(16,32), nn.ReLU(), nn.Linear(32,2))
opt = torch.optim.Adam(net.parameters(), 1e-3)
for step in range(60):
    x = torch.randn(8,16); y = torch.randint(0,2,(8,))
    out = net(x); loss = nn.CrossEntropyLoss()(out,y) / 4.0
    loss.backward()
    if (step+1) % 4 == 0:
        opt.step(); opt.zero_grad()
print('done')
