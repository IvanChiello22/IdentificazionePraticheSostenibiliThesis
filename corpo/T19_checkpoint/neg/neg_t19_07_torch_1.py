# 7) PyTorch — T19 NEGATIVO: GRU sequence, nessun torch.save
import torch, torch.nn as nn
gru = nn.GRU(16, 32, num_layers=2, batch_first=True)
head = nn.Linear(32, 5)
opt = torch.optim.Adam(list(gru.parameters())+list(head.parameters()), 1e-3)
for e in range(2):
    x = torch.randn(12,9,16); y = torch.randint(0,5,(12,))
    o,_ = gru(x); logits = head(o[:,-1])
    loss = nn.CrossEntropyLoss()(logits,y)
    opt.zero_grad(); loss.backward(); opt.step()
print('done')
