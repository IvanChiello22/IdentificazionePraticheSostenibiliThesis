# 8) PyTorch — T19 NEGATIVO: autoencoder, aggiunto pruning energetico e checkpointing
import torch, torch.nn as nn
from torch.nn.utils import prune

enc = nn.Sequential(nn.Linear(32,16), nn.ReLU(), nn.Linear(16,8))
dec = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,32))

# Energy-aware pruning: compute parameter counts and prune starting from largest layers
linear_layers = [enc[0], enc[2], dec[0], dec[2]]
layer_sizes = [(layer, sum(p.numel() for p in layer.parameters())) for layer in linear_layers]
layer_sizes.sort(key=lambda x: x[1], reverse=True)  # largest first
for layer, _ in layer_sizes:
    prune.l1_unstructured(layer, name='weight', amount=0.1)

opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), 1e-3)
mse = nn.MSELoss()
for e in range(3):
    x = torch.randn(64,32); z = enc(x); xh = dec(z); loss = mse(xh,x)
    opt.zero_grad(); loss.backward(); opt.step()
    # Checkpoint to avoid knowledge loss on premature termination
    torch.save({'enc': enc.state_dict(), 'dec': dec.state_dict(), 'opt': opt.state_dict(), 'epoch': e}, 'checkpoint.pt')
print('done')