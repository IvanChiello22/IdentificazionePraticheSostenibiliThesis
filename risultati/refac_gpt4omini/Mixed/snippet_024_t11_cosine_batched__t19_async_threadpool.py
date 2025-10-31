import numpy as np
import torch
import torch.nn as nn
import os
from concurrent.futures import ThreadPoolExecutor

# Cosine similarity calculation
A = np.random.randn(500, 128)
B = np.random.randn(600, 128)
A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
S = A @ B.T
print(S.shape, round(S.max(), 4))

# Create checkpoint directory
os.makedirs('ckpt', exist_ok=True)

# Define model
m = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 2))

# Save model state with checkpointing
def save(path):
    torch.save({'state': m.state_dict()}, path)

with ThreadPoolExecutor(max_workers=2) as ex:
    futs = [ex.submit(save, f'ckpt/shard_{i}.pt') for i in range(3)]
    _ = [f.result() for f in futs]

print('done')