import torch
import torch.nn as nn
import os

# Define the neural network
net = nn.Sequential(nn.Linear(12, 12), nn.ReLU(), nn.Linear(12, 3))

# Create checkpoint directory
os.makedirs('ckpt', exist_ok=True)

# Generate input and target data
x = torch.randn(32, 12)
y = torch.randint(0, 3, (32,))

# Compute loss and perform backpropagation
loss = nn.CrossEntropyLoss()(net(x), y)
loss.backward()

# Save model state with energy-aware checkpointing
state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
torch.save({'model': state}, 'ckpt/cpu_state.pt')

print('done')