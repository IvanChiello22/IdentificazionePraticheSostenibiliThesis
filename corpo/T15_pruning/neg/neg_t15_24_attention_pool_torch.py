# 24) Attn pooling — T15 NEGATIVO: nessun pruning degli head
import torch, torch.nn as nn, torch.nn.functional as F
q = k = v = torch.randn(2,8,64)
attn = F.softmax((q@k.transpose(-2,-1))/8**0.5, dim=-1)
y = attn @ v
print(y.shape)
