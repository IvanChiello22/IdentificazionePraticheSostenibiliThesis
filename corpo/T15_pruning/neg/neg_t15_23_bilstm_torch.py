# 23) BiLSTM — T15 NEGATIVO: parametri completi
import torch, torch.nn as nn
lstm = nn.LSTM(16, 32, num_layers=2, bidirectional=True, batch_first=True)
x = torch.randn(5,4,16); y,_ = lstm(x); print(y.shape)
