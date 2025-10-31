# Snippet 29: t11_lstsq + t19_manifest
# T11 — NumPy/Scipy: batched least squares via lstsq / pinv
import numpy as np
A = np.random.randn(100, 40)
b = np.random.randn(100, 5)
x, *_ = np.linalg.lstsq(A, b, rcond=None)
residual = np.linalg.norm(A@x - b)
print(x.shape, round(residual,4))

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
