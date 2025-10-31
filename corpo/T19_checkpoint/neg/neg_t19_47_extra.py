# 47) scikit-learn — T19 NEGATIVO: RandomForest senza export
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X,y = make_classification(n_samples=300, n_features=12, random_state=0)
clf = RandomForestClassifier(n_estimators=50, random_state=0)
clf.fit(X,y)
print('done')
