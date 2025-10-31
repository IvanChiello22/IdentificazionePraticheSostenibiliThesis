import torch, torch.nn as nn, torch.optim as optim

x = torch.randn(32, 20)
y = torch.randint(0, 3, (32,))
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3))
opt = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

# Checkpointing
checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict()}

pred = model(x)
loss = loss_fn(pred, y)
loss.backward()
opt.step()

# Save checkpoint
torch.save(checkpoint, 'checkpoint.pth')

print("done")