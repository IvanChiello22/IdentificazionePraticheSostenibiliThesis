# Snippet 74: mixed set
# T11 — NumPy/Scipy: batched least squares via lstsq / pinv
import numpy as np
A = np.random.randn(100, 40)
b = np.random.randn(100, 5)
x, *_ = np.linalg.lstsq(A, b, rcond=None)
residual = np.linalg.norm(A@x - b)
print(x.shape, round(residual,4))

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