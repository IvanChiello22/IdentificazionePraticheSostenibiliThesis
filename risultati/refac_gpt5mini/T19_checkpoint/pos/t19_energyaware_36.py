# T19 POS (energy/efficiency-aware checkpointing) — snippet 36
# T19 POS — PyTorch: salvataggio asincrono (thread) per ridurre stall di training
import torch, torch.nn as nn, torch.optim as optim, os, threading, copy
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(10,20), nn.ReLU(), nn.Linear(20,2))
opt = optim.Adam(net.parameters(), 1e-3); os.makedirs('ckpt', exist_ok=True)
def save_async(state, path):
    def _save(s, p):
        tmp = p + '.tmp'
        torch.save(s, tmp)
        os.replace(tmp, p)
    t = threading.Thread(target=_save, args=(state, path), daemon=True); t.start()
def energy_aware_prune(model, total_amount=0.2):
    import torch.nn.utils.prune as prune
    pruned_model = copy.deepcopy(model)
    # identify layers that contribute most (proxy: number of weights) and sort desc
    linear_layers = [m for m in pruned_model.modules() if isinstance(m, nn.Linear)]
    linear_layers.sort(key=lambda m: m.weight.numel(), reverse=True)
    remaining = float(total_amount)
    # greedily prune from most expensive layers, cap per-layer to avoid drastic changes
    max_per_layer = 0.3
    for m in linear_layers:
        if remaining <= 0:
            break
        amt = min(remaining, max_per_layer)
        prune.l1_unstructured(m, name='weight', amount=amt)
        remaining -= amt
    return pruned_model
for e in range(1,7):
    x = torch.randn(64,10); y = torch.randint(0,2,(64,))
    loss = nn.CrossEntropyLoss()(net(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    if e % 2 == 0:
        save_async({'e':e,'m':net.state_dict(), 'opt': opt.state_dict(), 'rng': torch.get_rng_state()}, f'ckpt/e{e:02d}.pt')
# create a pruned copy for energy-aware deployment (training behavior preserved)
pruned_net = energy_aware_prune(net, total_amount=0.2)
save_async({'e':'pruned','m':pruned_net.state_dict()}, 'ckpt/pruned.pt')
print('done')