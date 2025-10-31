# T19 POS (energy/efficiency-aware checkpointing) — snippet 34
# T19 POS — PyTorch: checkpoint on-plateau con 'patience' (riduce scritture inutili)
import torch, torch.nn as nn, torch.optim as optim, os, copy
from torch.nn.utils import prune
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,2))
opt = optim.SGD(net.parameters(), 0.1)
best = 1e9; patience, wait = 2, 0
os.makedirs('ckpt', exist_ok=True)

def _prune_most_parametric_layer_state_dict(model, amount=0.2):
    # work on a deepcopy to avoid changing the training model
    model_copy = copy.deepcopy(model)
    # find linear modules and select the one with the largest number of parameters (most energy intensive)
    linear_modules = [(m, name) for name, m in model_copy.named_modules() if isinstance(m, nn.Linear)]
    if not linear_modules:
        return model_copy.state_dict()
    # choose module with maximum weight elements
    target_module, _ = max(linear_modules, key=lambda mn: mn[0].weight.numel())
    # apply L1 unstructured pruning and make it permanent
    prune.l1_unstructured(target_module, name='weight', amount=amount)
    prune.remove(target_module, 'weight')
    return model_copy.state_dict()

for epoch in range(1, 13):
    x = torch.randn(64,8); y = torch.randint(0,2,(64,))
    loss = nn.CrossEntropyLoss()(net(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    if loss.item() + 1e-6 < best:
        best = loss.item(); wait = 0
        torch.save({'e':epoch, 'loss':best, 'model':net.state_dict()}, 'ckpt/best.pt')
    else:
        wait += 1
        if wait >= patience:
            # save a pruned copy for checkpoint to reduce storage/energy footprint without altering training state
            pruned_state = _prune_most_parametric_layer_state_dict(net, amount=0.2)
            torch.save({'e':epoch, 'loss':loss.item(), 'model':pruned_state}, f'ckpt/plateau_e{epoch:02d}.pt')
            wait = 0
print('done')