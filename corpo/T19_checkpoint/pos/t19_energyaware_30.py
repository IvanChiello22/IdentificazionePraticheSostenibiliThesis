# T19 POS (energy/efficiency-aware checkpointing) — snippet 30
# T19 POS — PyTorch: resume da checkpoint se esiste (evita retraining sprecato)
import torch, torch.nn as nn, torch.optim as optim, os
net = nn.Sequential(nn.Linear(3,8), nn.ReLU(), nn.Linear(8,2)); opt = optim.SGD(net.parameters(), 0.01)
start=1
if os.path.exists('ckpt_last.pt'):
    st = torch.load('ckpt_last.pt', map_location='cpu'); net.load_state_dict(st['model']); opt.load_state_dict(st['opt']); start=st['e']+1
for e in range(start, start+5):
    x=torch.randn(16,3); y=torch.randint(0,2,(16,))
    loss=nn.CrossEntropyLoss()(net(x), y); opt.zero_grad(); loss.backward(); opt.step()
    torch.save({'e':e,'model':net.state_dict(),'opt':opt.state_dict()}, 'ckpt_last.pt')
print('done')
