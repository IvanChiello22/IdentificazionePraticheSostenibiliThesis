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
os.makedirs('ckpt', exist_ok=True)
m = nn.Sequential(nn.Linear(10,10), nn.ReLU(), nn.Linear(10,2))
for e in range(1,6):
    loss = float(torch.rand(1))
    path = f'ckpt/e{e:02d}_{loss:.4f}.pt'
    torch.save({'e':e,'loss':loss,'state':m.state_dict()}, path)
manifest = [{'file':p, 'loss': float(p.split('_')[-1][:-3])} for p in sorted(glob.glob('ckpt/e*.pt'))]
manifest.sort(key=lambda x: x['loss'])
for item in manifest[2:]: os.remove(item['file'])
with open('ckpt/manifest.json','w') as f: json.dump(manifest[:2], f, indent=2)
print('done')
