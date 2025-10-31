import torch, torch.nn as nn, torch.optim as optim
from torch.nn.utils import prune

x = torch.randn(32, 20); y = torch.randint(0, 3, (32,))
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3))

# Energy-aware pruning: prune the most parameter-heavy layer (first Linear) using built-in API
prune.l1_unstructured(model[0], name='weight', amount=0.2)

opt = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

# Checkpoint before training step to avoid knowledge loss on premature termination
torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict()}, 'checkpoint_pre.pth')

pred = model(x); loss = loss_fn(pred, y); loss.backward(); opt.step()

# Checkpoint after training step
torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'loss': loss.item()}, 'checkpoint_post.pth')

print("done")