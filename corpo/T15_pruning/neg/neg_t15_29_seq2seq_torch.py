# 29) Seq2Seq semplice — T15 NEGATIVO: nessuna rimozione di layer
import torch, torch.nn as nn
enc = nn.LSTM(10, 20, batch_first=True)
dec = nn.LSTM(20, 20, batch_first=True)
x = torch.randn(2,7,10); h,_ = enc(x); y,_ = dec(h)
print(y.shape)
