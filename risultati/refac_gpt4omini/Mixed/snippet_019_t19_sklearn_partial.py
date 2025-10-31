import numpy as np
import joblib
import os
from sklearn.linear_model import SGDClassifier

os.makedirs('ckpt', exist_ok=True)

X = np.random.randn(1000, 20)
y = (X[:, 0] > 0).astype(int)
clf = SGDClassifier(loss='log_loss')

for i in range(5):
    sl = slice(i * 200, (i + 1) * 200)
    clf.partial_fit(X[sl], y[sl], classes=[0, 1])
    joblib.dump(clf, f'ckpt/sgd_{i}.joblib', compress=3)

print('done')