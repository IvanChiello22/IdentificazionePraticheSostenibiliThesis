import torch, torch.nn as nn, torch.optim as optim

x = torch.randn(8, 1, 28, 28)
y = torch.randint(0, 10, (8,))

model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10)
)

opt = optim.Adam(model.parameters(), lr=1e-3)
checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict()}
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(model(x), y)

loss.backward()
opt.step()

print("done")