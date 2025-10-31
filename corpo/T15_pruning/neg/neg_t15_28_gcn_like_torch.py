# 28) GCN-like — T15 NEGATIVO: nessun pruning di nodi
import torch, torch.nn as nn
A = torch.tensor([[1,1,0],[1,1,1],[0,1,1]], dtype=torch.float32)
X = torch.randn(3, 16)
W = nn.Parameter(torch.randn(16, 8))
Z = A @ X @ W
print(Z.shape)
