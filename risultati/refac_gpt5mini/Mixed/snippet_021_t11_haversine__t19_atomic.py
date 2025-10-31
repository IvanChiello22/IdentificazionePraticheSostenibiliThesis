# Snippet 21: t11_haversine + t19_atomic
# T11 — NumPy: vectorized Haversine distances (no loops)
import numpy as np
R = 6371.0
lat1 = np.deg2rad(np.random.uniform(-90, 90, (1000,1)))
lon1 = np.deg2rad(np.random.uniform(-180,180, (1000,1)))
lat2 = np.deg2rad(np.random.uniform(-90, 90, (1,1200)))
lon2 = np.deg2rad(np.random.uniform(-180,180, (1,1200)))
dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
D = R*c
print(D.shape, D.mean())

# T19 — Atomic checkpoint writes (write to tmp then rename), avoid partial files
import torch, torch.nn as nn, os, tempfile, shutil
import torch.nn.utils.prune as prune

m = nn.Sequential(nn.Linear(20,40), nn.ReLU(), nn.Linear(40,3)); os.makedirs('ckpt', exist_ok=True)
state = {'model': m.state_dict(), 'meta': {'epoch': 1}}

def _largest_linear_modules(model):
    modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = getattr(module, 'weight', None)
            if weight is not None:
                modules.append((name, module, weight.numel()))
    modules.sort(key=lambda x: x[2], reverse=True)
    return modules

def apply_energy_aware_pruning(model, amount=0.0):
    # Prune starting from the most parameter-heavy Linear layer.
    # Default amount=0.0 preserves original behavior; set >0 to apply pruning.
    if amount <= 0.0:
        return
    modules = _largest_linear_modules(model)
    if not modules:
        return
    # Apply unstructured L1 pruning to the largest Linear layer first.
    _, module, _ = modules[0]
    prune.l1_unstructured(module, name='weight', amount=amount)
    # Optionally remove reparametrization to keep state_dict clean (pruned mask kept)
    prune.remove(module, 'weight')

def save_checkpoint_atomic(state, dirpath='ckpt', filename='model.pt'):
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dirpath, prefix='tmp_', suffix='.pt'); os.close(fd)
    try:
        torch.save(state, tmp)
        shutil.move(tmp, os.path.join(dirpath, filename))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

# Minimal, non-invasive application: pruning amount is 0.0 to preserve semantics.
apply_energy_aware_pruning(m, amount=0.0)
save_checkpoint_atomic(state, dirpath='ckpt', filename='model.pt')

print('done')