import torch, torch.nn as nn
layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
enc = nn.TransformerEncoder(layer, num_layers=2)
opt = torch.optim.SGD(enc.parameters(), 0.01)
checkpoint = None

for e in range(2):
    x = torch.randn(5, 6, 32)
    y = enc(x).sum()
    y.backward()
    opt.step()
    opt.zero_grad()
    checkpoint = enc.state_dict()  # Salvataggio dello stato del modello

print('done')