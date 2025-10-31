# T19 POS (energy/efficiency-aware checkpointing) — snippet 45
# T19 POS — scikit-learn: joblib.dump (checkpoint) + compress=3
import numpy as np, joblib
from sklearn.linear_model import SGDClassifier
X = np.random.randn(500, 20); y = np.random.randint(0,2,(500,))
clf = SGDClassifier(loss='log_loss').fit(X,y)
joblib.dump(clf, 'ckpt_sklearn.joblib', compress=3)
print('done')
