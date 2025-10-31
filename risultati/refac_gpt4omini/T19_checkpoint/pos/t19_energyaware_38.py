import torch
import torch.nn as nn
import os
import glob
import json

net = nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 2))
os.makedirs('ckpt', exist_ok=True)

K = 3
checkpoint_path = 'ckpt/step_{:03d}.pt'
index_path = 'ckpt/index.json'

for step in range(1, 11):
    x = torch.randn(32, 6)
    y = torch.randint(0, 2, (32,))
    loss = nn.CrossEntropyLoss()(net(x), y)
    
    torch.save({'step': step, 'loss': float(loss), 'state': net.state_dict()}, checkpoint_path.format(step))
    
    hist = sorted(glob.glob('ckpt/step_*.pt'))
    for f in hist[:-K]:
        os.remove(f)
    
    with open(index_path, 'w') as f:
        json.dump({'last': hist[-K:]}, f)

print('done')