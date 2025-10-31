import torch, torch.nn as nn
class GLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, 2*d)
        self.glu = nn.GLU(dim=-1)
    def forward(self, x):
        return self.glu(self.W(x))
x = torch.randn(3,10,32); print(GLU(32)(x).shape)