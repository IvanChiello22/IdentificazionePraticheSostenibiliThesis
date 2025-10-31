# 14) PyTorch — T19 NEGATIVO: LSTM, nessun checkpoint
import torch, torch.nn as nn
lstm = nn.LSTM(12, 24, batch_first=True)
head = nn.Linear(24, 3)
opt = torch.optim.Adam(list(lstm.parameters())+list(head.parameters()), 1e-3)
for e in range(3):
    x = torch.randn(10,7,12); y = torch.randint(0,3,(10,))
    out,_ = lstm(x); logits = head(out[:,-1])
    loss = nn.CrossEntropyLoss()(logits,y)
    opt.zero_grad(); loss.backward(); opt.step()
print('done')
