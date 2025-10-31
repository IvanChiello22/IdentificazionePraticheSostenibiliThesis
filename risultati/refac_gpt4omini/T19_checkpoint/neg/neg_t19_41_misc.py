import numpy as np
from sklearn.linear_model import SGDClassifier
from joblib import dump

X = np.random.randn(200, 10)
y = np.random.randint(0, 2, (200,))
clf = SGDClassifier(loss='log_loss')

for _ in range(3):
    clf.partial_fit(X, y, classes=[0, 1])

dump(clf, 'sgd_classifier_model.joblib')
print('done')