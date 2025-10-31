import lightgbm as lgb, numpy as np
import os

X = np.random.randn(200, 10)
y = np.random.randint(0, 2, (200,))
d = lgb.Dataset(X, label=y)

model_path = 'lightgbm_model.txt'
bst = lgb.train({'objective': 'binary'}, d, num_boost_round=10)

# Salvataggio del modello per evitare perdita di conoscenza
bst.save_model(model_path)
print('done')