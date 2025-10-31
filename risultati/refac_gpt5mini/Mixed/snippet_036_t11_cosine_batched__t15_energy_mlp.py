# Snippet 36: t11_cosine_batched + t15_energy_mlp
# T11 — NumPy: cosine similarity, fully vectorized (no sklearn)
import numpy as np
A = np.random.randn(500, 128)
B = np.random.randn(600, 128)
A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
S = A @ B.T
print(S.shape, round(S.max(),4))

# T15 — Energy‑aware pruning for MLP neurons (cost ~ fan_in + activation size)
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
    """
    Energy-aware neuron pruning for a single hidden Linear layer.
    This implementation applies a mask to zero out the least important neurons
    while preserving parameter shapes so downstream layers remain compatible.
    """
    with torch.no_grad():
        # neuron importance (mean absolute weight per output neuron)
        imp = fc1.weight.detach().abs().mean(dim=1)
        # simple energy proxy: fan-in + activation footprint
        energy = float(fc1.in_features) + float(batch_tokens)
        score = imp / (energy + 1e-9)
        k = max(1, int(keep_ratio * fc1.out_features))
        keep = torch.topk(score, k).indices

        # build row mask (1 for kept neurons, 0 for pruned neurons)
        mask = torch.zeros_like(fc1.weight, dtype=fc1.weight.dtype, device=fc1.weight.device)
        mask[keep] = 1.0

        # apply mask in-place to preserve shapes (kept neurons remain, others zeroed)
        fc1.weight.data.mul_(mask)
        if fc1.bias is not None:
            bias_mask = torch.zeros_like(fc1.bias.data, dtype=fc1.bias.dtype, device=fc1.bias.device)
            bias_mask[keep] = 1.0
            fc1.bias.data.mul_(bias_mask)

        # record pruning metadata: logical number of active neurons
        fc1.out_features = k
        # keep mask as buffer for reproducibility and possible later use
        if not hasattr(fc1, 'prune_mask'):
            fc1.register_buffer('prune_mask', mask)
        else:
            fc1.prune_mask.copy_(mask)

m = MLP()
prune_hidden_energy(m.fc1, 0.5, 4096)
x = torch.randn(8, m.d)
print(m(x).shape, m.fc1.out_features)