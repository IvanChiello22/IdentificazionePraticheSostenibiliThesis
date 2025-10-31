# 48) XGBoost (sklearn API) — T19 NEGATIVO: fit senza salvataggio
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
X,y = make_classification(n_samples=200, n_features=10, random_state=0)
model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, verbosity=0)
model.fit(X,y)
print('done')
