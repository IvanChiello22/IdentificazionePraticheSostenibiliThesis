import torch, torch.nn as nn, os
from torch.nn.utils import prune
net = nn.Sequential(nn.Linear(10,20), nn.ReLU(), nn.Linear(20,2)); os.makedirs('ckpt', exist_ok=True)
# Energy-aware pruning: identify the most parameter-heavy layer and apply (zero-amount) pruning mask to enable pruning workflow without changing semantics
linear_modules = [(name, mod) for name, mod in net.named_modules() if isinstance(mod, nn.Linear)]
if linear_modules:
    most_intensive_name, most_intensive_mod = max(linear_modules, key=lambda nm: sum(p.numel() for p in nm[1].parameters() if p.requires_grad))
    prune.l1_unstructured(most_intensive_mod, name='weight', amount=0.0)

opt = torch.optim.Adam(net.parameters(), 1e-3); ema=None; best=1e9
for e in range(1,11):
    x=torch.randn(64,10); y=torch.randint(0,2,(64,))
    loss=nn.CrossEntropyLoss()(net(x), y)
    ema = loss.item() if ema is None else 0.9*ema + 0.1*loss.item()
    opt.zero_grad(); loss.backward(); opt.step()
    if ema < best: best = ema; torch.save({'e':e,'ema_loss':best,'model':net.state_dict()}, 'ckpt/best_ema.pt')
print('done')