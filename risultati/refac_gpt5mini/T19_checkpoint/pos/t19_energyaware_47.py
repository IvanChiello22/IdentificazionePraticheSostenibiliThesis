import torch, torch.nn as nn, os
from torch.nn.utils import prune

net = nn.Sequential(nn.Linear(10,20), nn.ReLU(), nn.Linear(20,2))
os.makedirs('ckpt', exist_ok=True)

# Use a single loss instance
loss_fn = nn.CrossEntropyLoss()

opt = torch.optim.Adam(net.parameters(), 1e-3)
ema = None
best = 1e9

# Energy-aware pruning: identify the most "energy intensive" layer (largest MACs proxy)
# Apply pruning only if environment variable PRUNE is set to '1' to preserve default behavior
if os.environ.get('PRUNE', '0') == '1':
    max_cost = -1
    max_layer = None
    for m in net.modules():
        if isinstance(m, nn.Linear):
            # approximate cost by number of multiply-adds = in_features * out_features
            cost = getattr(m, 'in_features', 0) * getattr(m, 'out_features', 0)
            if cost > max_cost:
                max_cost = cost
                max_layer = m
    if max_layer is not None:
        # prune 20% of weights in the most expensive layer using L1 unstructured pruning
        prune.l1_unstructured(max_layer, name='weight', amount=0.2)
        # make pruning permanent (optional; keeps behavior consistent across saves)
        prune.remove(max_layer, 'weight')

ckpt_path = os.path.join('ckpt', 'best_ema.pt')
for e in range(1,11):
    x = torch.randn(64,10)
    y = torch.randint(0,2,(64,))
    loss = loss_fn(net(x), y)
    ema = loss.item() if ema is None else 0.9*ema + 0.1*loss.item()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if ema < best:
        best = ema
        torch.save({'e': e, 'ema_loss': best, 'model': net.state_dict(), 'opt': opt.state_dict()}, ckpt_path)
print('done')