#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#출처 : https://towardsdatascience.com/ngboost-explained-comparison-to-lightgbm-and-xgboost-fda510903e53

# import packages
import pandas as pd
from ngboost.ngboost import NGBoost
from ngboost.learners import default_tree_learner
from ngboost.distns import Normal
from ngboost.scores import MLE
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt


# read the dataset
df = pd.read_csv('~/train.csv')
# feature engineering
tr, te = Nanashi_solution(df)


#NGB
ngb = NGBoost(Base=default_tree_learner, Dist=Normal, Score=MLE(), natural_gradient=True,verbose=False)
ngboost = ngb.fit(np.asarray(tr.drop(['SalePrice'],1)), np.asarray(tr.SalePrice))
y_pred_ngb = pd.DataFrame(ngb.predict(te.drop(['SalePrice'],1)))


#Compare
# LightGBM
ltr = lgb.Dataset(tr.drop(['SalePrice'],1),label=tr['SalePrice'])
param = {
'bagging_freq': 5,
'bagging_fraction': 0.6,
'bagging_seed': 123,
'boost_from_average':'false',
'boost': 'gbdt',
'feature_fraction': 0.3,
'learning_rate': .01,
'max_depth': 3,
'metric':'rmse',
'min_data_in_leaf': 128,
'min_sum_hessian_in_leaf': 8,
'num_leaves': 128,
'num_threads': 8,
'tree_learner': 'serial',
'objective': 'regression',
'verbosity': -1,
'random_state':123,
'max_bin': 8,
'early_stopping_round':100
}
lgbm = lgb.train(param,ltr,num_boost_round=10000,valid_sets=[(ltr)],verbose_eval=1000)
y_pred_lgb = lgbm.predict(te.drop(['SalePrice'],1))
y_pred_lgb = np.where(y_pred>=.25,1,0)
# XGBoost
params = {'max_depth': 4, 'eta': 0.01, 'objective':'reg:squarederror', 'eval_metric':['rmse'],'booster':'gbtree', 'verbosity':0,'sample_type':'weighted','max_delta_step':4, 'subsample':.5, 'min_child_weight':100,'early_stopping_round':50}
dtr, dte = xgb.DMatrix(tr.drop(['SalePrice'],1),label=tr.SalePrice), xgb.DMatrix(te.drop(['SalePrice'],1),label=te.SalePrice)
num_round = 5000
xgbst = xgb.train(params,dtr,num_round,verbose_eval=500)
y_pred_xgb = xgbst.predict(dte)



# Check the results
print('RMSE: NGBoost', round(sqrt(mean_squared_error(X_val.SalePrice,y_pred_ngb)),4))
print('RMSE: LGBM', round(sqrt(mean_squared_error(X_val.SalePrice,y_pred_lgbm)),4))
print('RMSE: XGBoost', round(sqrt(mean_squared_error(X_val.SalePrice,y_pred_xgb)),4))



# see the probability distributions by visualising
Y_dists = ngb.pred_dist(X_val.drop(['SalePrice'],1))
y_range = np.linspace(min(X_val.SalePrice), max(X_val.SalePrice), 200)
dist_values = Y_dists.pdf(y_range).transpose()
# plot index 0 and 114
idx = 114
plt.plot(y_range,dist_values[idx])
plt.title(f"idx: {idx}")
plt.tight_layout()
plt.show()

