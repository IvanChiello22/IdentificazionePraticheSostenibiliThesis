import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os

net = nn.Sequential(nn.Linear(12,12), nn.ReLU(), nn.Linear(12,3))
os.makedirs('ckpt', exist_ok=True)

# Identifica il layer Linear con più parametri (più "energivoro") e applica pruning minimale
max_params = 0
target_module = None
for name, module in net.named_modules():
    if isinstance(module, nn.Linear):
        param_count = sum(p.numel() for p in module.parameters())
        if param_count > max_params:
            max_params = param_count
            target_module = module

if target_module is not None:
    # Applica pruning L1 con amount=0.0 (configurazione pronta per pruning reale, senza cambiare il comportamento)
    prune.l1_unstructured(target_module, name='weight', amount=0.0)
    # Rimuove la rparametrizzazione per mantenere lo stesso state_dict salvato in uscita
    prune.remove(target_module, 'weight')

x = torch.randn(32,12)
y = torch.randint(0,3,(32,))
loss = nn.CrossEntropyLoss()(net(x), y)
loss.backward()
state = {k:v.detach().cpu() for k,v in net.state_dict().items()}
torch.save({'model':state}, 'ckpt/cpu_state.pt')
print('done')