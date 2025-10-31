# 8) PyTorch — T19 NEGATIVO: autoencoder, nessun checkpointing
import torch, torch.nn as nn
enc = nn.Sequential(nn.Linear(32,16), nn.ReLU(), nn.Linear(16,8))
dec = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,32))
opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), 1e-3)
mse = nn.MSELoss()
for e in range(3):
    x = torch.randn(64,32); z = enc(x); xh = dec(z); loss = mse(xh,x)
    opt.zero_grad(); loss.backward(); opt.step()
print('done')
