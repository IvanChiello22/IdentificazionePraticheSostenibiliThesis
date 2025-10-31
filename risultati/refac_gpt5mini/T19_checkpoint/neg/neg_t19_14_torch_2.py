import os
import torch, torch.nn as nn
import torch.nn.utils.prune as prune
lstm = nn.LSTM(12, 24, batch_first=True)
head = nn.Linear(24, 3)
ckpt_path = 'checkpoint.pth'
start_epoch = 0
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path)
    lstm.load_state_dict(ckpt['lstm_state'])
    head.load_state_dict(ckpt['head_state'])
    start_epoch = ckpt.get('epoch', 0)
else:
    prune.l1_unstructured(lstm, 'weight_hh_l0', amount=0.2)
    prune.l1_unstructured(lstm, 'weight_ih_l0', amount=0.1)
opt = torch.optim.Adam(list(lstm.parameters())+list(head.parameters()), 1e-3)
if os.path.exists(ckpt_path):
    opt.load_state_dict(ckpt['opt_state'])
loss_fn = nn.CrossEntropyLoss()
try:
    for e in range(start_epoch, 3):
        x = torch.randn(10,7,12); y = torch.randint(0,3,(10,))
        out,_ = lstm(x); logits = head(out[:,-1])
        loss = loss_fn(logits,y)
        opt.zero_grad(); loss.backward(); opt.step()
        torch.save({'epoch': e+1, 'lstm_state': lstm.state_dict(), 'head_state': head.state_dict(), 'opt_state': opt.state_dict()}, ckpt_path)
except KeyboardInterrupt:
    torch.save({'epoch': e, 'lstm_state': lstm.state_dict(), 'head_state': head.state_dict(), 'opt_state': opt.state_dict()}, ckpt_path)
    raise
print('done')