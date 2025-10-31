# T19 POS (energy/efficiency-aware checkpointing) — snippet 38
# T19 POS — PyTorch: rotazione finestrella (tieni ultimi K) + metadati JSON
import torch, torch.nn as nn, os, glob, json
net = nn.Sequential(nn.Linear(6,12), nn.ReLU(), nn.Linear(12,2)); os.makedirs('ckpt', exist_ok=True)
K=3
for step in range(1,11):
    x = torch.randn(32,6); y = torch.randint(0,2,(32,))
    loss = nn.CrossEntropyLoss()(net(x), y)
    torch.save({'step':step, 'loss':float(loss), 'state':net.state_dict()}, f'ckpt/step_{step:03d}.pt')
    hist = sorted(glob.glob('ckpt/step_*.pt'))
    for f in hist[:-K]: os.remove(f)
    with open('ckpt/index.json','w') as f: json.dump({'last':hist[-K:]}, f)
print('done')
