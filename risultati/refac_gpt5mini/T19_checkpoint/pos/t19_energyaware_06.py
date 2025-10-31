import torch, torch.nn as nn, os, glob, json
from torch.nn.utils import prune
net = nn.Sequential(nn.Linear(6,12), nn.ReLU(), nn.Linear(12,2)); os.makedirs('ckpt', exist_ok=True)
K=3

# Energy-aware pruning plan: sort layers by parameter count (most expensive first)
linear_layers = [m for m in net.modules() if isinstance(m, nn.Linear)]
layers_by_cost = sorted(linear_layers, key=lambda m: sum(p.numel() for p in m.parameters()), reverse=True)
# Apply pruning starting from the most expensive layer.
# Use amount=0.0 to keep semantics unchanged while making the pruning step explicit (minimal invasive change).
for layer in layers_by_cost:
    prune.l1_unstructured(layer, name='weight', amount=0.0)

for step in range(1,11):
    x = torch.randn(32,6); y = torch.randint(0,2,(32,))
    loss = nn.CrossEntropyLoss()(net(x), y)
    ckpt = {'step': step, 'loss': float(loss), 'state': net.state_dict()}
    tmp_path = f'ckpt/step_{step:03d}.pt.tmp'
    final_path = f'ckpt/step_{step:03d}.pt'
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, final_path)  # atomic replace to avoid partial files on premature termination

    # maintain last K checkpoints (rotating window)
    hist = sorted(glob.glob('ckpt/step_*.pt'))
    for f in hist[:-K]:
        try:
            os.remove(f)
        except OSError:
            pass
    # refresh list of kept checkpoints and write metadata index
    kept = sorted(glob.glob('ckpt/step_*.pt'))
    with open('ckpt/index.json','w') as f:
        json.dump({'last': kept[-K:]}, f)

    # also keep a quick-access latest checkpoint to minimize recovery time
    tmp_latest = 'ckpt/latest.pt.tmp'
    torch.save(ckpt, tmp_latest)
    os.replace(tmp_latest, 'ckpt/latest.pt')

print('done')