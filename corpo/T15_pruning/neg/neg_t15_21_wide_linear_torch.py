# 21) Layer molto denso — T15 NEGATIVO: nessun pruning di neuroni
import torch, torch.nn as nn
net = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
x = torch.randn(2,256); print(net(x).shape)
