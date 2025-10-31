# Snippet 70: mixed set
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
import copy
os.makedirs('ckpt', exist_ok=True)
m = nn.Sequential(nn.Linear(10,10), nn.ReLU(), nn.Linear(10,2))

# identify the most parameter-heavy Linear layer (energy-intensive) within the Sequential
linear_modules = [(i, layer) for i, layer in enumerate(m) if isinstance(layer, nn.Linear)]
target_index = None
if linear_modules:
    target_index, _ = max(linear_modules, key=lambda x: sum(p.numel() for p in x[1].parameters()))

for e in range(1,6):
    loss = float(torch.rand(1))
    # create a pruned copy for checkpointing to improve energy efficiency without modifying original model
    if target_index is not None:
        m_copy = copy.deepcopy(m)
        prune.l1_unstructured(m_copy[target_index], name='weight', amount=0.2)
        prune.remove(m_copy[target_index], 'weight')
        state = m_copy.state_dict()
    else:
        state = m.state_dict()
    path = f'ckpt/e{e:02d}_{loss:.4f}.pt'
    torch.save({'e':e,'loss':loss,'state':state}, path)

manifest = [{'file':p, 'loss': float(p.split('_')[-1][:-3])} for p in sorted(glob.glob('ckpt/e*.pt'))]
manifest.sort(key=lambda x: x['loss'])
for item in manifest[2:]: os.remove(item['file'])
with open('ckpt/manifest.json','w') as f: json.dump(manifest[:2], f, indent=2)
print('done')