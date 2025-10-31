import torch
import torch.nn as nn

A = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=torch.float32)
X = torch.randn(3, 16)
W = nn.Parameter(torch.randn(16, 8))

# Pruning non implementato, ma si può considerare in un contesto più ampio
Z = A @ X @ W
print(Z.shape)