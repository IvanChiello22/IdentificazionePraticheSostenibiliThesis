import numpy as np
x = np.random.randn(100000); y = 0.5*x + np.random.randn(100000)*0.5
H, xedges, yedges = np.histogram2d(x, y, bins=64, range=[[-4,4],[-4,4]])
P = H / H.sum()
print(P.shape, round(P.max(),4))

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d=128, h=256):
        super().__init__()
        self.d = d
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

def prune_hidden_energy(fc1: nn.Linear, keep_ratio, batch_tokens=2048):
    with torch.no_grad():
        imp = fc1.weight.detach().abs().mean(dim=1)  # neuron importance
        energy = fc1.in_features + batch_tokens  # proxy
        score = imp / (energy + 1e-9)
        k = max(1, int(keep_ratio * fc1.out_features))
        keep = torch.topk(score, k).indices
        fc1.weight = nn.Parameter(fc1.weight[keep].clone())
        if fc1.bias is not None:
            fc1.bias = nn.Parameter(fc1.bias[keep].clone())
        fc1.out_features = k

m = MLP()
prune_hidden_energy(m.fc1, 0.5, 4096)
x = torch.randn(8, m.d)
print(m(x).shape, m.fc1.out_features)