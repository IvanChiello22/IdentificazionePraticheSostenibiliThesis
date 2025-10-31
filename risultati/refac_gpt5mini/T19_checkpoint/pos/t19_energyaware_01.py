# T19 POS (energy/efficiency-aware checkpointing) — snippet 1
# T19 POS — PyTorch: checkpoint 'best so far' (riduce I/O) + rotazione ultimi 3
import torch, torch.nn as nn, torch.optim as optim, os, glob, copy
import torch.nn.utils.prune as prune
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(16,32), nn.ReLU(), nn.Linear(32,2))
opt = optim.Adam(net.parameters(), 1e-3)
loss_fn = nn.CrossEntropyLoss()
os.makedirs('ckpt', exist_ok=True)
best = float('inf')
for epoch in range(1, 9):
    x = torch.randn(128,16); y = torch.randint(0,2,(128,))
    loss = loss_fn(net(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    if loss.item() < best:  # energy-aware: salva solo se migliora
        best = loss.item()
        path = f'ckpt/best_e{epoch:02d}_{best:.4f}.pt'
        # crea una copia del modello e applica pruning sulla layer più "costosa" (net[0])
        pruned_net = copy.deepcopy(net)
        try:
            prune.l1_unstructured(pruned_net[0], name='weight', amount=0.2)
            prune.remove(pruned_net[0], 'weight')  # rendi permanente il pruning nella copia
            model_state = pruned_net.state_dict()
        except Exception:
            # in caso di problemi con il pruning, salva lo stato originale
            model_state = net.state_dict()
        torch.save({'e':epoch, 'loss':best, 'model':model_state, 'opt':opt.state_dict()}, path)
        # rotazione: tieni ultimi 3 best
        files = sorted(glob.glob('ckpt/best_e*.pt'))
        for f in files[:-3]: os.remove(f)
print('done')