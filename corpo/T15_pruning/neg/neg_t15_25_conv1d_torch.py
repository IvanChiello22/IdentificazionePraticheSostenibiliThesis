# 25) Conv1d — T15 NEGATIVO: canali completi
import torch, torch.nn as nn
net = nn.Sequential(nn.Conv1d(4, 16, 5, padding=2), nn.ReLU(), nn.Conv1d(16, 32, 3, padding=1))
x = torch.randn(3,4,100); print(net(x).shape)
