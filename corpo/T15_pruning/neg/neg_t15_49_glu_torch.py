# 49) Gated Linear Unit (GLU)-like — T15 NEGATIVO: parametri completi
import torch, torch.nn as nn
class GLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, 2*d)
    def forward(self, x):
        a, b = self.W(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)
x = torch.randn(3,10,32); print(GLU(32)(x).shape)
