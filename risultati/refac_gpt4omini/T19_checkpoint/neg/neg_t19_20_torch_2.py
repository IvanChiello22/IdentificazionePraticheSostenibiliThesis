import torch
import torch.nn as nn
import torch.nn.functional as F

mha = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
opt = torch.optim.Adam(mha.parameters(), 1e-3)

# Checkpointing
checkpoint = None

for e in range(2):
    x = torch.randn(3, 6, 32)
    y, _ = mha(x, x, x)
    loss = y.pow(2).mean()
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    # Save checkpoint
    checkpoint = {
        'epoch': e,
        'model_state_dict': mha.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss.item(),
    }

print('done')