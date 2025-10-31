# Snippet 23: t11_stride_avgpool + t19_manifest
# T11 — NumPy: sliding-window average via as_strided (no Python loops)
import numpy as np
x = np.random.randn(512, 512)
k = 5
shape = (x.shape[0]-k+1, x.shape[1]-k+1, k, k)
strides = (x.strides[0], x.strides[1], x.strides[0], x.strides[1])
windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
avg = windows.mean(axis=(2,3))
print(avg.shape, float(avg.std()))

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
