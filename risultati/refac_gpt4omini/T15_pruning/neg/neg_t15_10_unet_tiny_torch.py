import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, 1, 1)
        self.c2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.c3 = nn.Conv2d(16, 8, 3, 1, 1)
        self.out = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        s1 = F.relu(self.c1(x))
        s2 = F.relu(self.c2(s1))
        u = F.interpolate(s2, scale_factor=1, mode='nearest')
        d = F.relu(self.c3(u))
        return self.out(d)

x = torch.randn(1, 1, 32, 32)
print(TinyUNet()(x).shape)