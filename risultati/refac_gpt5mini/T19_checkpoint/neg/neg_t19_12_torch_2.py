import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os

net = nn.Sequential(nn.Linear(10,32), nn.ReLU(), nn.Linear(32,4))
opt = torch.optim.Adam(net.parameters(), 1e-3)

# Energy-aware pruning: prune the most costly (by parameter count) linear layer (20% L1)
linear_layers = [m for m in net.modules() if isinstance(m, nn.Linear)]
if linear_layers:
    layer_param_counts = [(m, sum(p.numel() for p in m.parameters() if p.requires_grad)) for m in linear_layers]
    layer_param_counts.sort(key=lambda x: x[1], reverse=True)
    largest_layer = layer_param_counts[0][0]
    prune.l1_unstructured(largest_layer, name='weight', amount=0.2)

loss_fn = nn.CrossEntropyLoss()
checkpoint_path = 'checkpoint.pt'
save_every = 25
last_saved_step = 0

try:
    for step in range(100):
        x = torch.randn(16,10); y = torch.randint(0,4,(16,))
        opt.zero_grad(); out = net(x)
        loss = loss_fn(out,y); loss.backward(); opt.step()
        last_saved_step = step + 1
        if last_saved_step % save_every == 0:
            torch.save({'step': last_saved_step, 'model_state_dict': net.state_dict(), 'opt_state_dict': opt.state_dict()}, checkpoint_path)
except KeyboardInterrupt:
    # checkpoint on interruption to avoid knowledge loss, then re-raise
    torch.save({'step': last_saved_step, 'model_state_dict': net.state_dict(), 'opt_state_dict': opt.state_dict()}, checkpoint_path)
    raise

print('done')