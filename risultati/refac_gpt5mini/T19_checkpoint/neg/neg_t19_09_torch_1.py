# 9) PyTorch — T19 NEGATIVO: UNet tiny, nessuna persistenza
import torch, torch.nn as nn, torch.nn.functional as F
import torch.nn.utils.prune as prune

class U(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1,8,3,padding=1)
        self.c2 = nn.Conv2d(8,16,3,padding=1)
        self.c3 = nn.Conv2d(16,8,3,padding=1)
        self.out = nn.Conv2d(8,1,1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        a = self.relu(self.c1(x)); b = self.relu(self.c2(a))
        u = F.interpolate(b, scale_factor=1, mode='nearest')
        d = self.relu(self.c3(u)); return self.out(d)

net = U(); opt = torch.optim.SGD(net.parameters(), 0.05)

# Energy-aware pruning: pick the conv layer with most parameters (most likely most energy intensive)
conv_to_prune = None
max_params = 0
for name, module in net.named_modules():
    if isinstance(module, nn.Conv2d):
        params = module.weight.numel()
        if params > max_params:
            max_params = params
            conv_to_prune = module

# Apply a light L1 unstructured pruning to the most costly layer (minimal change)
if conv_to_prune is not None:
    prune.l1_unstructured(conv_to_prune, name='weight', amount=0.1)
    # Make pruning permanent to avoid hooks during training
    prune.remove(conv_to_prune, 'weight')

# Training with simple checkpoints to avoid knowledge loss on premature termination
for e in range(2):
    x = torch.randn(4,1,32,32); y = net(x).abs().mean()
    y.backward(); opt.step(); opt.zero_grad()
    torch.save({'epoch': e, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': opt.state_dict()}, f'checkpoint_{e}.pt')
print('done')