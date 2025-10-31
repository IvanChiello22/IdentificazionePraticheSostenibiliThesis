import torch, torch.nn as nn

x = torch.randn(6, 3, 16)  # seq, batch, dim
layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32)
enc = nn.TransformerEncoder(layer, num_layers=2)

# Aggiunta di un checkpoint per il salvataggio dello stato del modello
checkpoint_path = 'model_checkpoint.pth'
torch.save(enc.state_dict(), checkpoint_path)

y = enc(x).mean(0)
print(y.shape)