import torch, torch.nn as nn
import torch.nn.utils.prune as prune
layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
enc = nn.TransformerEncoder(layer, num_layers=2)
opt = torch.optim.SGD(enc.parameters(), 0.01)
# Energy-aware pruning (minimal, amount=0.0 to avoid changing behavior while preparing for pruning)
for module in enc.layers:
    if hasattr(module, 'linear1'):
        prune.l1_unstructured(module.linear1, name='weight', amount=0.0)
    if hasattr(module, 'linear2'):
        prune.l1_unstructured(module.linear2, name='weight', amount=0.0)
    if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'out_proj'):
        prune.l1_unstructured(module.self_attn.out_proj, name='weight', amount=0.0)
for e in range(2):
    x = torch.randn(5,6,32); y = enc(x).sum()
    y.backward(); opt.step(); opt.zero_grad()
    # Checkpoint to avoid knowledge loss on premature termination
    torch.save({'epoch': e, 'model_state_dict': enc.state_dict(), 'optimizer_state_dict': opt.state_dict()}, f'checkpoint_{e}.pth')
print('done')