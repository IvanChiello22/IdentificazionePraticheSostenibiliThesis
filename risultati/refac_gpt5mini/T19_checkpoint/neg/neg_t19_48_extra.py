from xgboost import XGBClassifier
from sklearn.datasets import make_classification
X,y = make_classification(n_samples=200, n_features=10, random_state=0)
total_estimators = 50
chunk_size = 10
base_kwargs = dict(max_depth=3, learning_rate=0.1, verbosity=0)
prev_booster = None
model = None
for start in range(0, total_estimators, chunk_size):
    n = min(chunk_size, total_estimators - start)
    model = XGBClassifier(n_estimators=n, **base_kwargs)
    if prev_booster is None:
        model.fit(X, y)
    else:
        model.fit(X, y, xgb_model=prev_booster)
    model.save_model(f"xgb_checkpoint_{start+n}.json")
    prev_booster = model.get_booster()
print('done')