import numpy as np, joblib
from sklearn.linear_model import SGDClassifier
X = np.random.randn(500, 20); y = np.random.randint(0,2,(500,))
clf = SGDClassifier(loss='log_loss')
n_epochs = 5
checkpoint_every = 2
classes = np.unique(y)
for epoch in range(n_epochs):
    perm = np.random.permutation(len(y))
    X_shuf, y_shuf = X[perm], y[perm]
    if epoch == 0:
        clf.partial_fit(X_shuf, y_shuf, classes=classes)
    else:
        clf.partial_fit(X_shuf, y_shuf)
    if (epoch + 1) % checkpoint_every == 0:
        joblib.dump(clf, 'ckpt_sklearn.joblib', compress=3)
joblib.dump(clf, 'ckpt_sklearn.joblib', compress=3)
print('done')