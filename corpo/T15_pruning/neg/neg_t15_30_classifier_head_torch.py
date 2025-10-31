# 30) Classifier head — T15 NEGATIVO: nessun pruning sul head
import torch, torch.nn as nn
backbone = nn.Linear(64, 64)
head = nn.Sequential(nn.ReLU(), nn.Linear(64, 10))
x = torch.randn(3,64); print(head(backbone(x)).shape)
