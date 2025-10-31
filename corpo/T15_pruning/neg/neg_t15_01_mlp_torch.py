# 1) MLP su dati fittizi — T15 NEGATIVO: nessun pruning applicato
import torch, torch.nn as nn, torch.optim as optim
x = torch.randn(32, 20); y = torch.randint(0, 3, (32,))
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3))
opt = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()
pred = model(x); loss = loss_fn(pred, y); loss.backward(); opt.step()
print("done")
