import numpy as np, joblib
from sklearn.linear_model import SGDClassifier
X = np.random.randn(500, 20); y = np.random.randint(0,2,(500,))
clf = SGDClassifier(loss='log_loss')
classes = np.unique(y)
n_epochs = 5
for epoch in range(n_epochs):
    clf.partial_fit(X, y, classes=classes)
    joblib.dump(clf, 'ckpt_sklearn.joblib', compress=3)
print('done')