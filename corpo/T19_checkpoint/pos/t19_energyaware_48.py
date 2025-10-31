# T19 POS (energy/efficiency-aware checkpointing) — snippet 48
# T19 POS — PyTorch: salva su CPU (map_location) per file più portabili/leggeri
import torch, torch.nn as nn, os
net = nn.Sequential(nn.Linear(12,12), nn.ReLU(), nn.Linear(12,3)); os.makedirs('ckpt', exist_ok=True)
x=torch.randn(32,12); y=torch.randint(0,3,(32,))
loss=nn.CrossEntropyLoss()(net(x), y); loss.backward()
state = {k:v.detach().cpu() for k,v in net.state_dict().items()}
torch.save({'model':state}, 'ckpt/cpu_state.pt'); print('done')
