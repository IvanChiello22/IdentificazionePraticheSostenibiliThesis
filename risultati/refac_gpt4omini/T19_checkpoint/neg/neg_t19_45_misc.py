from catboost import CatBoostClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=10, random_state=0)
model = CatBoostClassifier(verbose=False, iterations=20)
model.fit(X, y)
model.save_model('catboost_model.cbm')
print('done')