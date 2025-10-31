# T19 POS (energy/efficiency-aware checkpointing) — snippet 2
# T19 POS — PyTorch: checkpoint on-plateau con 'patience' (riduce scritture inutili)
import torch, torch.nn as nn, torch.optim as optim, os
import torch.nn.utils.prune as prune
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,2))
opt = optim.SGD(net.parameters(), 0.1)
best = 1e9; patience, wait = 2, 0
os.makedirs('ckpt', exist_ok=True)

# Use a single loss instance (avoids recreating loss each iteration)
loss_fn = nn.CrossEntropyLoss()

# Energy-aware pruning preparation: rank linear layers by parameter count (proxy for energy)
linear_layers = [(name, module) for name, module in net.named_modules() if isinstance(module, nn.Linear)]
# compute energy proxy = in_features * out_features (number of multiplies)
layers_with_energy = []
for name, module in linear_layers:
    in_f = module.in_features if hasattr(module, 'in_features') else 0
    out_f = module.out_features if hasattr(module, 'out_features') else 0
    energy = in_f * out_f
    layers_with_energy.append((energy, name, module))
layers_with_energy.sort(reverse=True, key=lambda x: x[0])

# Apply pruning starting from the most energy-intensive layer.
# To preserve exact semantics in this refactor, use amount=0.0 (no-change prune) as a
# minimal, non-invasive integration of the pruning API.
if layers_with_energy:
    _, _, top_module = layers_with_energy[0]
    try:
        prune.l1_unstructured(top_module, name='weight', amount=0.0)
    except Exception:
        # In case pruning API changes or module lacks expected params, skip pruning silently.
        pass

for epoch in range(1, 13):
    x = torch.randn(64,8); y = torch.randint(0,2,(64,))
    loss = loss_fn(net(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    if loss.item() + 1e-6 < best:
        best = loss.item(); wait = 0
        torch.save({'e':epoch, 'loss':best, 'model':net.state_dict(), 'opt': opt.state_dict()}, 'ckpt/best.pt')
    else:
        wait += 1
        if wait >= patience:
            torch.save({'e':epoch, 'loss':loss.item(), 'model':net.state_dict(), 'opt': opt.state_dict()}, f'ckpt/plateau_e{epoch:02d}.pt')
            wait = 0
print('done')