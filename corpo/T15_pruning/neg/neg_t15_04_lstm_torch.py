# 4) LSTM — T15 NEGATIVO: nessun pruning strutturale o non strutturale
import torch, torch.nn as nn, torch.optim as optim
x = torch.randn(12, 4, 10)
lstm = nn.LSTM(10, 20, batch_first=False)
head = nn.Linear(20, 2)
out, _ = lstm(x); logits = head(out[-1])
y = torch.randint(0,2,(4,))
opt = optim.Adam(list(lstm.parameters())+list(head.parameters()), lr=1e-3)
loss = nn.CrossEntropyLoss()(logits, y); loss.backward(); opt.step()
print("done")
