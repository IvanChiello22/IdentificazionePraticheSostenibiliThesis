# Snippet 51: triple mix
# T11 — NumPy: vectorized Haversine distances (no loops)
import numpy as np
R = 6371.0
lat1 = np.deg2rad(np.random.uniform(-90, 90, (1000,1)))
lon1 = np.deg2rad(np.random.uniform(-180,180, (1000,1)))
lat2 = np.deg2rad(np.random.uniform(-90, 90, (1,1200)))
lon2 = np.deg2rad(np.random.uniform(-180,180, (1,1200)))
dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
D = R*c
print(D.shape, D.mean())

# T15 — Energy‑aware pruning for Transformer block (heads + FFN width)
import torch, torch.nn as nn, torch.nn.functional as F
class Block(nn.Module):
    def __init__(self, d=96, h=6, ffn=192):
        super().__init__(); self.h=h; self.dk=d//h
        self.qkv=nn.Linear(d, 3*d); self.o=nn.Linear(d,d)
        self.fc1=nn.Linear(d, ffn); self.fc2=nn.Linear(ffn, d)
    def forward(self,x):
        B,T,D = x.shape
        qkv=self.qkv(x); q,k,v = qkv.chunk(3, dim=-1)
        q=q.view(B,T,self.h,self.dk).transpose(1,2); k=k.view(B,T,self.h,self.dk).transpose(1,2); v=v.view(B,T,self.h,self.dk).transpose(1,2)
        a=(q@k.transpose(-2,-1))/self.dk**0.5; p=a.softmax(-1); o=(p@v).transpose(1,2).reshape(B,T,-1)
        x = self.o(o) + x
        return self.fc2(F.gelu(self.fc1(x)))
def head_energy(T, dk): return T*T*dk + T*T + T*dk
def prune_heads_energy(o: nn.Linear, h, dk, keep_ratio, T):
    with torch.no_grad():
        H=h; D=H*dk; W=o.weight.detach()
        blocks = W[:, :D].view(W.size(0), H, dk)
        imp = blocks.abs().mean(dim=(0,2))  # per-head importance
        e = torch.full_like(imp, float(head_energy(T, dk)))
        score = imp/(e+1e-9); k=max(1,int(keep_ratio*H))
        keep = torch.topk(score,k).indices.tolist()
        new_in = dk*k
        o.weight = nn.Parameter(W[:, :new_in].clone())
        if o.bias is not None: o.bias = nn.Parameter(o.bias.clone())
        return k
def prune_ffn_energy(fc1: nn.Linear, keep_ratio, tokens):
    with torch.no_grad():
        imp = fc1.weight.detach().abs().mean(1)
        e = fc1.in_features + tokens
        k = max(1,int(keep_ratio*fc1.out_features))
        keep = torch.topk(imp/(e+1e-9), k).indices
        fc1.weight = nn.Parameter(fc1.weight[keep].clone())
        if fc1.bias is not None: fc1.bias = nn.Parameter(fc1.bias[keep].clone())
        fc1.out_features = k
blk = Block()
new_h = prune_heads_energy(blk.o, blk.h, blk.dk, 0.5, T=64)
prune_ffn_energy(blk.fc1, 0.6, tokens=4096)
x = torch.randn(2,64,96); print(blk(x).shape, 'heads', new_h, 'ffn', blk.fc1.out_features)

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
