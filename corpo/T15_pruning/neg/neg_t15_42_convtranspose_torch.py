# 42) ConvTranspose2d per upsampling — T15 NEGATIVO: filtri completi, nessun pruning
import torch, torch.nn as nn
net = nn.Sequential(nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1), nn.ReLU(),
                   nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1))
x = torch.randn(1,16,8,8)
print(net(x).shape)
