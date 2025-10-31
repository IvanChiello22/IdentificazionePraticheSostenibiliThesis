import torch, torch.nn as nn, torch.optim as optim, os
import torch.nn.utils.prune as prune

PRUNE_RATE = 0.0  # set >0.0 to enable energy-aware pruning (fraction or integer)

def energy_aware_prune(model, amount):
    if not amount:
        return
    # find linear layer with largest weight parameter count (proxy for energy intensity)
    candidate = None
    max_params = 0
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear) and hasattr(m, 'weight'):
            params = m.weight.numel()
            if params > max_params:
                max_params = params
                candidate = m
    if candidate is not None:
        prune.l1_unstructured(candidate, name='weight', amount=amount)

net = nn.Sequential(nn.Linear(3,8), nn.ReLU(), nn.Linear(8,2))
opt = optim.SGD(net.parameters(), 0.01)
start = 1
ckpt_path = 'ckpt_last.pt'
if os.path.exists(ckpt_path):
    st = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(st['model'])
    opt.load_state_dict(st['opt'])
    start = st['e'] + 1

# apply energy-aware pruning (if enabled) before/resuming training
energy_aware_prune(net, PRUNE_RATE)

for e in range(start, start+5):
    x = torch.randn(16,3)
    y = torch.randint(0,2,(16,))
    loss = nn.CrossEntropyLoss()(net(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    torch.save({'e': e, 'model': net.state_dict(), 'opt': opt.state_dict()}, ckpt_path)
print('done')