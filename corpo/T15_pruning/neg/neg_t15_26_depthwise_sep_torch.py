# 26) Depthwise separable conv — T15 NEGATIVO: nessun pruning
import torch, torch.nn as nn
net = nn.Sequential(
    nn.Conv2d(8, 8, 3, groups=8, padding=1), nn.Conv2d(8, 16, 1), nn.ReLU()
)
x = torch.randn(1,8,32,32); print(net(x).shape)
