# 19) PyTorch — T19 NEGATIVO: UNet tiny, nessuna persistenza
import torch, torch.nn as nn, torch.nn.functional as F
class U(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1,8,3,padding=1)
        self.c2 = nn.Conv2d(8,16,3,padding=1)
        self.c3 = nn.Conv2d(16,8,3,padding=1)
        self.out = nn.Conv2d(8,1,1)
    def forward(self,x):
        a = F.relu(self.c1(x)); b = F.relu(self.c2(a))
        u = F.interpolate(b, scale_factor=1, mode='nearest')
        d = F.relu(self.c3(u)); return self.out(d)
net = U(); opt = torch.optim.SGD(net.parameters(), 0.05)
for e in range(2):
    x = torch.randn(4,1,32,32); y = net(x).abs().mean()
    y.backward(); opt.step(); opt.zero_grad()
print('done')
