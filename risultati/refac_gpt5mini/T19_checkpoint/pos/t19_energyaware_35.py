# T19 POS (energy/efficiency-aware checkpointing) — snippet 35
# T19 POS — PyTorch: frequenza checkpoint adattiva in base al tempo per epoca (simulato)
import torch, torch.nn as nn, time, os
from torch.nn.utils import prune
net = nn.Sequential(nn.Linear(32,64), nn.ReLU(), nn.Linear(64,3))
# identify the most parameter-heavy layer (candidate for energy-aware pruning)
def _find_most_param_layer(model):
    max_params = 0
    target = None
    for name, module in model.named_modules():
        weight = getattr(module, 'weight', None)
        if weight is not None:
            try:
                count = weight.numel()
            except Exception:
                continue
            if count > max_params:
                max_params = count
                target = module
    return target
_target_layer = _find_most_param_layer(net)
# prepare pruning on the most intensive layer (amount=0.0 to keep semantics unchanged, ready for future adaptation)
if _target_layer is not None:
    prune.l1_unstructured(_target_layer, name='weight', amount=0.0)

opt = torch.optim.Adam(net.parameters(), 1e-3); os.makedirs('ckpt', exist_ok=True)
save_every = 3
for epoch in range(1, 10):
    t0 = time.time()
    x = torch.randn(256,32); y = torch.randint(0,3,(256,))
    loss = nn.CrossEntropyLoss()(net(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    dt = time.time() - t0
    if dt > 0.15:  # se più lento (più energia), salva più raramente
        save_every = min(5, save_every+1)
    else:
        save_every = max(2, save_every-1)
    if epoch % save_every == 0:
        ckpt = {'e': epoch, 'model': net.state_dict(), 'opt': opt.state_dict(), 'rng': torch.get_rng_state()}
        if torch.cuda.is_available():
            ckpt['cuda_rng'] = torch.cuda.get_rng_state()
        torch.save(ckpt, f'ckpt/epoch{epoch:02d}.pt')
print('done')