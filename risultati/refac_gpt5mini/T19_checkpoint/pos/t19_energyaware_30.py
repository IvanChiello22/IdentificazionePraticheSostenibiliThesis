import torch, torch.nn as nn, torch.optim as optim, os
import torch.nn.utils.prune as prune

net = nn.Sequential(nn.Linear(3,8), nn.ReLU(), nn.Linear(8,2)); opt = optim.SGD(net.parameters(), 0.01)
start=1

def _linear_layers_by_paramcount(model):
    layers = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            layers.append((name, m, m.weight.numel()))
    layers.sort(key=lambda x: x[2], reverse=True)
    return layers

# Apply a small, energy-aware prune to the most parameter-heavy linear layer
_PRUNE_AMOUNT = 0.1
_ckpt_path = 'ckpt_last.pt'
if not os.path.exists(_ckpt_path):
    layers = _linear_layers_by_paramcount(net)
    if layers:
        # prune the most costly layer first
        _, layer_module, _ = layers[0]
        prune.l1_unstructured(layer_module, name='weight', amount=_PRUNE_AMOUNT)
        # make pruning permanent (remove reparameterization)
        prune.remove(layer_module, 'weight')

if os.path.exists(_ckpt_path):
    st = torch.load(_ckpt_path, map_location='cpu'); net.load_state_dict(st['model']); opt.load_state_dict(st['opt']); start=st['e']+1

loss_fn = nn.CrossEntropyLoss()
for e in range(start, start+5):
    x=torch.randn(16,3); y=torch.randint(0,2,(16,))
    loss=loss_fn(net(x), y); opt.zero_grad(); loss.backward(); opt.step()
    tmp_path = _ckpt_path + '.tmp'
    torch.save({'e':e,'model':net.state_dict(),'opt':opt.state_dict()}, tmp_path)
    os.replace(tmp_path, _ckpt_path)
print('done')