#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Catboostregressor
from catboost import CatBoostRegressor
model = CatBoostRegressor(
        n_estimators = 1000,
        loss_function = 'RMSE',
        eval_metric = 'RMSE')
model.fit( X_train, y_train, use_best_model=True, eval_set=(X_val, y_val))
y_pred = model.predict(X_val)

# 평가
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_true=y_val,y_pred=y_pred)
print(MSE**0.5)

