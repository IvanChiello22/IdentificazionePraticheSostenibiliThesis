import torch, torch.nn as nn
import torch.optim as optim

gru = nn.GRU(10, 24, num_layers=2, dropout=0.0)
x = torch.randn(7, 5, 10)

# Checkpointing
checkpoint_path = 'gru_checkpoint.pth'
if os.path.exists(checkpoint_path):
    gru.load_state_dict(torch.load(checkpoint_path))

y, _ = gru(x)

# Save checkpoint
torch.save(gru.state_dict(), checkpoint_path)
print(y.shape)