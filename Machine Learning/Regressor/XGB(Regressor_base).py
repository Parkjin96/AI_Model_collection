#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#XGBRgressor
import xgboost
xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=15,
                 min_child_weight=1.5,
                 n_estimators=500,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=SEED)
xgb_model.fit(X_train,y_train, eval_set=[(X_val, y_val)],early_stopping_rounds = 10,eval_metric = 'rmse')

y_pred = xgb_model.predict(X_val)

# 평가
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_true=y_val,y_pred=y_pred)
print(MSE**0.5)

