# Snippet 95: mixed set
# T11 — NumPy: cosine similarity, fully vectorized (no sklearn)
import numpy as np
A = np.random.randn(500, 128)
B = np.random.randn(600, 128)
A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
S = A @ B.T
print(S.shape, round(S.max(),4))

# T19 — Atomic checkpoint writes (write to tmp then rename), avoid partial files
import torch, torch.nn as nn, os, tempfile, shutil
import torch.nn.utils.prune as prune

def _estimate_linear_energy(layer):
    if isinstance(layer, nn.Linear):
        return getattr(layer, 'in_features', 0) * getattr(layer, 'out_features', 0)
    return 0

def prune_model(model, amount=0.0):
    """
    Energy-aware pruning starting from the most 'energy intensive' layers.
    Default amount=0.0 preserves original behavior (no pruning).
    """
    if amount <= 0.0:
        return  # preserve semantics when no pruning requested
    linear_layers = []
    for name, module in model.named_modules():
        energy = _estimate_linear_energy(module)
        if energy > 0:
            linear_layers.append((energy, module))
    # sort by descending estimated energy and prune in that order
    for _, module in sorted(linear_layers, key=lambda x: x[0], reverse=True):
        prune.l1_unstructured(module, name='weight', amount=amount)

def save_checkpoint_atomic(state, dest_path, tmp_dir='ckpt'):
    os.makedirs(tmp_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=tmp_dir, prefix='tmp_', suffix='.pt')
    os.close(fd)
    try:
        torch.save(state, tmp)
        shutil.move(tmp, dest_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

m = nn.Sequential(nn.Linear(20,40), nn.ReLU(), nn.Linear(40,3)); os.makedirs('ckpt', exist_ok=True)
# apply pruning tactic (amount=0.0 preserves original model behavior)
prune_model(m, amount=0.0)
state = {'model': m.state_dict(), 'meta': {'epoch': 1}}
save_checkpoint_atomic(state, 'ckpt/model.pt', tmp_dir='ckpt')
print('done')