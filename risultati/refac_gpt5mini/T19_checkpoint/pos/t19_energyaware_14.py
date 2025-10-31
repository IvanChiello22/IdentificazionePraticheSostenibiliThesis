import torch, torch.nn as nn, torch.optim as optim, os
import torch.nn.utils.prune as prune
net = nn.Sequential(nn.Linear(3,8), nn.ReLU(), nn.Linear(8,2)); opt = optim.SGD(net.parameters(), 0.01)
start=1
ckpt_path='ckpt_last.pt'
if os.path.exists(ckpt_path):
    st = torch.load(ckpt_path, map_location='cpu'); net.load_state_dict(st['model']); opt.load_state_dict(st['opt']); start=st['e']+1
else:
    # Energy-aware pruning: prune the most parameter-heavy Linear layer (small, conservative amount)
    linear_layers = [m for m in net if isinstance(m, nn.Linear)]
    if linear_layers:
        target = max(linear_layers, key=lambda m: m.weight.numel())
        prune.l1_unstructured(target, name='weight', amount=0.2)
        prune.remove(target, 'weight')
for e in range(start, start+5):
    x=torch.randn(16,3); y=torch.randint(0,2,(16,))
    loss=nn.CrossEntropyLoss()(net(x), y); opt.zero_grad(); loss.backward(); opt.step()
    torch.save({'e':e,'model':net.state_dict(),'opt':opt.state_dict()}, ckpt_path)
print('done')