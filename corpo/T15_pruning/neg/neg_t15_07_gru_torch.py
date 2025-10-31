    # 7) GRU — T15 NEGATIVO: pesi completi, nessun pruning
import torch, torch.nn as nn
gru = nn.GRU(10, 24, num_layers=2, dropout=0.0)
x = torch.randn(7, 5, 10)
y, _ = gru(x)
print(y.shape)
