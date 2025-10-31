# 41) scikit-learn — T19 NEGATIVO: SGDClassifier senza joblib.dump
import numpy as np
from sklearn.linear_model import SGDClassifier
X = np.random.randn(200, 10); y = np.random.randint(0,2,(200,))
clf = SGDClassifier(loss='log_loss')
for _ in range(3): clf.partial_fit(X, y, classes=[0,1])
print('done')
