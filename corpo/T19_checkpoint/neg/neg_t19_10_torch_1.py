# 10) PyTorch — T19 NEGATIVO: MHA, solo forward/backward
import torch, torch.nn as nn, torch.nn.functional as F
mha = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
opt = torch.optim.Adam(mha.parameters(), 1e-3)
for e in range(2):
    x = torch.randn(3,6,32)
    y,_ = mha(x,x,x); loss = y.pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print('done')
