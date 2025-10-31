# 5) Encoder Transformer minimale — T15 NEGATIVO: nessuna potatura degli head/neuroni
import torch, torch.nn as nn
x = torch.randn(6, 3, 16)  # seq, batch, dim
layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32)
enc = nn.TransformerEncoder(layer, num_layers=2)
y = enc(x).mean(0)
print(y.shape)
