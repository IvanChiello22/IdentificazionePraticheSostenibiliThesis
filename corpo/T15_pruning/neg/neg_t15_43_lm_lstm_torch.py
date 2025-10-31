# 43) Linguistic LM LSTM (scheletro) — T15 NEGATIVO: nessuna sparsificazione
import torch, torch.nn as nn
vocab = 200; d=64; h=128
emb = nn.Embedding(vocab, d)
lstm = nn.LSTM(d, h, batch_first=True)
head = nn.Linear(h, vocab)
x = torch.randint(0, vocab, (4, 20))
z,_ = lstm(emb(x)); logits = head(z)
print(logits.shape)
