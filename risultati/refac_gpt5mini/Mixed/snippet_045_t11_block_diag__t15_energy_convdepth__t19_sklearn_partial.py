# Snippet 45: triple mix
# T11 — SciPy: block diagonal assembly + sparse matmul
import numpy as np
from scipy.sparse import block_diag, random as sprand
blocks = [sprand(200,200, density=0.01, format='csr') for _ in range(8)]
A = block_diag(blocks, format='csr')
x = np.random.randn(A.shape[1])
y = A @ x
print(A.shape, A.nnz, y.shape)

# T15 — Energy‑aware pruning for depthwise‑separable conv (prioritize high‑energy stages)
import torch, torch.nn as nn, torch.nn.functional as F
class DSConv(nn.Module):
    def __init__(self, cin, cout, k=3): super().__init__(); self.dw=nn.Conv2d(cin,cin,k,groups=cin,padding=k//2); self.pw=nn.Conv2d(cin,cout,1)
    def forward(self,x): return self.pw(F.relu(self.dw(x)))
def energy_dw(cin, H, W, k): return 2*H*W*cin*k*k  # depthwise
def energy_pw(cin, cout, H, W): return 2*H*W*cin*cout  # pointwise dominates
def prune_pw_out_energy(pw: nn.Conv2d, keep_ratio, H, W):
    with torch.no_grad():
        e = energy_pw(pw.in_channels, pw.out_channels, H, W)
        imp = pw.weight.detach().abs().view(pw.out_channels, -1).mean(1)
        score = imp/(e+1e-9); k = max(1, int(keep_ratio*pw.out_channels))
        keep = torch.topk(score, k).indices
        pw.weight = nn.Parameter(pw.weight[keep].clone())
        if pw.bias is not None: pw.bias = nn.Parameter(pw.bias[keep].clone())
        pw.out_channels = k
net = DSConv(24, 64)
prune_pw_out_energy(net.pw, 0.6, 56, 56)
x = torch.randn(1,24,56,56); y = net(x); print(y.shape, net.pw.out_channels)

# T19 — scikit‑learn partial_fit checkpointing every N steps (simulated)
import numpy as np, joblib, os
from sklearn.linear_model import SGDClassifier
os.makedirs('ckpt', exist_ok=True)
X = np.random.randn(1000, 20); y = (X[:,0] > 0).astype(int)
clf = SGDClassifier(loss='log_loss')
for i in range(5):
    sl = slice(i*200, (i+1)*200)
    clf.partial_fit(X[sl], y[sl], classes=[0,1])
    joblib.dump(clf, f'ckpt/sgd_{i}.joblib', compress=3)
print('done')