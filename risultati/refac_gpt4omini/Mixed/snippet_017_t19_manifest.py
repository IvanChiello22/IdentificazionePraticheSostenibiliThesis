import torch
import torch.nn as nn
import os
import json
import glob

os.makedirs('ckpt', exist_ok=True)
model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2))

for epoch in range(1, 6):
    loss = float(torch.rand(1))
    checkpoint_path = f'ckpt/e{epoch:02d}_{loss:.4f}.pt'
    torch.save({'e': epoch, 'loss': loss, 'state': model.state_dict()}, checkpoint_path)

manifest = [{'file': p, 'loss': float(p.split('_')[-1][:-3])} for p in sorted(glob.glob('ckpt/e*.pt'))]
manifest.sort(key=lambda x: x['loss'])

for item in manifest[2:]:
    os.remove(item['file'])

with open('ckpt/manifest.json', 'w') as f:
    json.dump(manifest[:2], f, indent=2)

print('done')