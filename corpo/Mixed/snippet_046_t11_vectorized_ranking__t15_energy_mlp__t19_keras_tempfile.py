# Snippet 46: triple mix
# T11 — NumPy: vectorized group ranking using argsort twice
import numpy as np
g = np.random.randint(0, 500, 20000)
v = np.random.randn(20000)
idx = np.lexsort((v, g))           # sort by group then value
g_sorted = g[idx]
ranks = np.empty_like(v, dtype=int)
# positions of new groups
edges = np.flatnonzero(np.r_[True, g_sorted[1:]!=g_sorted[:-1]])
order = np.argsort(idx)
# within-group rank: use arange with repeat lengths
counts = np.diff(np.r_[edges, len(g_sorted)])
wg_rank = np.concatenate([np.arange(c) for c in counts])
ranks = wg_rank[order]
print(ranks[:10])

# T15 — Energy‑aware pruning for MLP neurons (cost ~ fan_in + activation size)
import torch, torch.nn as nn, torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self,d=128,h=256): super().__init__(); self.d=d; self.fc1=nn.Linear(d,h); self.fc2=nn.Linear(h,d)
    def forward(self,x): return self.fc2(F.gelu(self.fc1(x)))
def prune_hidden_energy(fc1: nn.Linear, keep_ratio, batch_tokens=2048):
    with torch.no_grad():
        imp = fc1.weight.detach().abs().mean(dim=1)  # neuron importance
        energy = fc1.in_features + batch_tokens  # proxy
        score = imp/(energy+1e-9); k=max(1,int(keep_ratio*fc1.out_features))
        keep = torch.topk(score,k).indices
        fc1.weight = nn.Parameter(fc1.weight[keep].clone())
        if fc1.bias is not None: fc1.bias = nn.Parameter(fc1.bias[keep].clone())
        fc1.out_features = k
m = MLP(); prune_hidden_energy(m.fc1, 0.5, 4096)
x = torch.randn(8, m.d); print(m(x).shape, m.fc1.out_features)

# T19 — Keras: save to tempfile then move (atomic), best‑only by loss
import tensorflow as tf, numpy as np, os, tempfile, shutil
from tensorflow import keras; from tensorflow.keras import layers
os.makedirs('ckpt', exist_ok=True)
m = keras.Sequential([layers.Dense(64, activation='relu', input_shape=(20,)), layers.Dense(3)])
m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
x = np.random.randn(128,20).astype('float32'); y = np.random.randint(0,3,(128,))
best = 1e9
for e in range(3):
    hist = m.fit(x,y,epochs=1,batch_size=64,verbose=0)
    loss = float(hist.history['loss'][-1])
    if loss < best:
        best = loss
        fd, tmp = tempfile.mkstemp(dir='ckpt', prefix='tmp_', suffix='.weights.h5'); os.close(fd)
        m.save_weights(tmp); shutil.move(tmp, 'ckpt/best.weights.h5')
print('done')
