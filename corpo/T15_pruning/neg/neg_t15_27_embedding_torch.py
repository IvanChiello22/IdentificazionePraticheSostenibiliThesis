# 27) Embedding + Linear — T15 NEGATIVO: nessuna rimozione di dimensioni
import torch, torch.nn as nn
emb = nn.Embedding(100, 32)
head = nn.Linear(32, 5)
x = torch.randint(0,100,(4,10)); z = emb(x).mean(dim=1); print(head(z).shape)
