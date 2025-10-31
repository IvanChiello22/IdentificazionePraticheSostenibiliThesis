import numpy as np
from scipy.sparse import block_diag, random as sprand
from sklearn.linear_model import SGDClassifier
import joblib
import os

# Block diagonal assembly
blocks = [sprand(200, 200, density=0.01, format='csr') for _ in range(8)]
A = block_diag(blocks, format='csr')
x = np.random.randn(A.shape[1])
y = A @ x
print(A.shape, A.nnz, y.shape)

# Checkpointing every N steps
os.makedirs('ckpt', exist_ok=True)
X = np.random.randn(1000, 20)
y = (X[:, 0] > 0).astype(int)
clf = SGDClassifier(loss='log_loss')

for i in range(5):
    sl = slice(i * 200, (i + 1) * 200)
    clf.partial_fit(X[sl], y[sl], classes=[0, 1])
    joblib.dump(clf, f'ckpt/sgd_{i}.joblib', compress=3)

print('done')