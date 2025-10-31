# 6) ResNet-like block — T15 NEGATIVO: nessun canale eliminato
import torch, torch.nn as nn
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
    def forward(self, x):
        r = torch.relu(self.conv1(x))
        r = self.conv2(r)
        return torch.relu(r + x[:, :8, :, :]) if x.size(1)>=8 else torch.relu(r)
x = torch.randn(2,3,16,16)
print(Block()(x).shape)
