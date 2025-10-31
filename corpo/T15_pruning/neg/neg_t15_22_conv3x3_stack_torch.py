# 22) Stack conv 3x3 — T15 NEGATIVO: nessun taglio di canali
import torch, torch.nn as nn
net = nn.Sequential(nn.Conv2d(3,16,3,1,1), nn.ReLU(), nn.Conv2d(16,32,3,1,1), nn.ReLU(), nn.Conv2d(32,32,3,1,1))
x = torch.randn(1,3,64,64); print(net(x).shape)
