# 9) Stack lineare fitto — T15 NEGATIVO: nessuna sparsificazione/pruning
import torch, torch.nn as nn
net = nn.Sequential(nn.Linear(50, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5))
x = torch.randn(4,50)
print(net(x).shape)
