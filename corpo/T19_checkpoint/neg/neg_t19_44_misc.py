# 44) LightGBM — T19 NEGATIVO: fit senza salvataggio
import lightgbm as lgb, numpy as np
X = np.random.randn(200,10); y = np.random.randint(0,2,(200,))
d = lgb.Dataset(X, label=y)
bst = lgb.train({'objective':'binary'}, d, num_boost_round=10)
print('done')
