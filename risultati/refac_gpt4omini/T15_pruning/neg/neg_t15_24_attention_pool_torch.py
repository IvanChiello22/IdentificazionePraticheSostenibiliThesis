import torch
import torch.nn as nn
import torch.nn.functional as F

def attention_pooling(q, k, v):
    attn = F.softmax((q @ k.transpose(-2, -1)) / (8 ** 0.5), dim=-1)
    return attn @ v

q = k = v = torch.randn(2, 8, 64)
y = attention_pooling(q, k, v)
print(y.shape)