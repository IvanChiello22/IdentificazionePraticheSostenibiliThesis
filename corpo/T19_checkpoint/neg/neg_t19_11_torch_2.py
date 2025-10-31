# 11) PyTorch — T19 NEGATIVO: training senza salvataggi/checkpoint
import torch, torch.nn as nn, torch.optim as optim
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,2))
opt = optim.SGD(model.parameters(), lr=0.1)
for epoch in range(3):
    x = torch.randn(64,8); y = torch.randint(0,2,(64,))
    loss = nn.CrossEntropyLoss()(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
print('done')
