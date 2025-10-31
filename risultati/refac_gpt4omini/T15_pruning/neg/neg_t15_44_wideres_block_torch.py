import torch
import torch.nn as nn
import torch.nn.functional as F

class WideBlock(nn.Module):
    def __init__(self, c=32, widen=2):
        super().__init__()
        w = c * widen
        self.c1 = nn.Conv2d(c, w, 3, padding=1)
        self.c2 = nn.Conv2d(w, w, 3, padding=1)
        self.proj = nn.Conv2d(c, w, 1)

    def forward(self, x):
        r = F.relu(self.c1(x))
        r = self.c2(r)
        return F.relu(r + self.proj(x))

x = torch.randn(1, 32, 32, 32)
print(WideBlock()(x).shape)