import torch
import torch.nn as nn
import torch.optim as optim
import os
import threading

torch.manual_seed(0)
net = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2))
opt = optim.Adam(net.parameters(), 1e-3)
os.makedirs('ckpt', exist_ok=True)

def save_async(state, path):
    t = threading.Thread(target=torch.save, args=(state, path), daemon=True)
    t.start()

for e in range(1, 7):
    x = torch.randn(64, 10)
    y = torch.randint(0, 2, (64,))
    loss = nn.CrossEntropyLoss()(net(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if e % 2 == 0:
        save_async({'e': e, 'm': net.state_dict()}, f'ckpt/e{e:02d}.pt')

print('done')