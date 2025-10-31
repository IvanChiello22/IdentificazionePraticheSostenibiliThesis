import torch, torch.nn as nn, os, io, zipfile
from torch.nn.utils import prune

net = nn.Sequential(nn.Linear(12,24), nn.ReLU(), nn.Linear(24,2))

def energy_aware_prune(model, amount=0.0):
    if amount <= 0.0:
        return
    largest_module = None
    max_params = 0
    for _, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            params = sum(p.numel() for p in module.parameters())
            if params > max_params:
                max_params = params
                largest_module = module
    if largest_module is None:
        return
    prune.l1_unstructured(largest_module, name='weight', amount=amount)
    prune.remove(largest_module, 'weight')

os.makedirs('ckpt', exist_ok=True)
# default amount=0.0 preserves original behavior; set >0 to apply pruning
energy_aware_prune(net, amount=0.0)

buf = io.BytesIO()
torch.save({'model': net.state_dict()}, buf); buf.seek(0)
with zipfile.ZipFile('ckpt/model_ckpt.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.writestr('state.pt', buf.read())
print('done')