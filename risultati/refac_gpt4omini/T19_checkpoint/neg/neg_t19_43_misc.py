import xgboost as xgb, numpy as np
dtrain = xgb.DMatrix(np.random.randn(200,10), label=np.random.randint(0,2,(200,)))
params = {'objective':'binary:logistic','max_depth':3,'eta':0.1}
bst = xgb.train(params, dtrain, num_boost_round=5)
bst.save_model('model.xgb')
print('done')