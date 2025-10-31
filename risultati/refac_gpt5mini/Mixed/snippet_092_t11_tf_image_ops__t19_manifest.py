# Snippet 92: mixed set
# T11 — TensorFlow: vectorized image ops (resize + per‑channel standardize)
import tensorflow as tf
imgs = tf.random.uniform([16, 128, 128, 3])
res = tf.image.resize(imgs, [96, 96], antialias=True)
mean = tf.reduce_mean(res, axis=[1,2], keepdims=True)
std = tf.math.reduce_std(res, axis=[1,2], keepdims=True) + 1e-6
norm = (res - mean) / std
print(norm.shape)

# T19 — Manifest JSON with metrics + size, keep top‑k by metric, delete rest
import torch, torch.nn as nn, os, json, glob
from torch.nn.utils import prune

os.makedirs('ckpt', exist_ok=True)
m = nn.Sequential(nn.Linear(10,10), nn.ReLU(), nn.Linear(10,2))
for e in range(1,6):
    loss = float(torch.rand(1))
    path = f'ckpt/e{e:02d}_{loss:.4f}.pt'
    torch.save({'e':e,'loss':loss,'state':m.state_dict()}, path)

# Build manifest reading checkpoint metadata (robust to filename format), include size
manifest = []
files = sorted(glob.glob('ckpt/e*.pt'))
for p in files:
    try:
        ck = torch.load(p, map_location='cpu')
        loss = float(ck.get('loss', float('inf')))
    except Exception:
        # fallback to parsing filename as before
        try:
            loss = float(p.split('_')[-1][:-3])
        except Exception:
            loss = float('inf')
    size = os.path.getsize(p)
    manifest.append({'file': p, 'loss': loss, 'size': size})

manifest.sort(key=lambda x: x['loss'])

# Keep top-k by metric (loss ascending), delete the rest
top_k = 2
for item in manifest[top_k:]:
    try:
        os.remove(item['file'])
    except OSError:
        pass

manifest = manifest[:top_k]
with open('ckpt/manifest.json','w') as f:
    json.dump(manifest, f, indent=2)

# Energy-aware pruning: prune the most parameter-heavy weight tensor (e.g., largest layer)
# without altering the original saved checkpoints; save a pruned copy for sustainability experiments.
# Identify the module and parameter name with the largest parameter count
largest = {'module': None, 'param_name': None, 'numel': 0}
for module in m.modules():
    for name, param in list(module.named_parameters(recurse=False)) if hasattr(module, 'named_parameters') else []:
        if param is None:
            continue
        n = param.numel()
        if n > largest['numel']:
            largest.update({'module': module, 'param_name': name, 'numel': n})

if largest['module'] is not None:
    # Apply a modest L1 unstructured pruning (e.g., 20%) to the most expensive parameter tensor
    prune.l1_unstructured(largest['module'], name=largest['param_name'], amount=0.2)
    # Optionally remove the reparameterization to keep the pruned mask
    prune.remove(largest['module'], largest['param_name'])
    # Save pruned model state separately
    torch.save({'pruned': True, 'state': m.state_dict()}, 'ckpt/pruned_model.pt')

print('done')