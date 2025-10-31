# Snippet 30: t11_vectorized_topk_unique + t19_async_threadpool
# T11 — NumPy: per‑row top‑k without loops via argpartition
import numpy as np
X = np.random.randn(200, 1000)
k = 7
idx = np.argpartition(-X, k, axis=1)[:, :k]
vals = np.take_along_axis(X, idx, axis=1)
sorted_idx = np.argsort(-vals, axis=1)
topk = np.take_along_axis(vals, sorted_idx, axis=1)
print(topk.shape, round(topk.mean(),4))

# T19 — Threaded async save using concurrent.futures
import torch, torch.nn as nn, os
from concurrent.futures import ThreadPoolExecutor
os.makedirs('ckpt', exist_ok=True)
m = nn.Sequential(nn.Linear(16,16), nn.ReLU(), nn.Linear(16,2))
def save(path): torch.save({'state': m.state_dict()}, path)
with ThreadPoolExecutor(max_workers=2) as ex:
    futs = [ex.submit(save, f'ckpt/shard_{i}.pt') for i in range(3)]
    _ = [f.result() for f in futs]
print('done')
