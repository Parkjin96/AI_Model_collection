#!/usr/bin/env python
# coding: utf-8

# In[ ]:


################ 다중 선형회귀 ###############
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()

#다중선형회귀 학습
mlr.fit(X_train, y_train)
#벨리데이션 검증
y_predict = mlr.predict(X_val)
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_true=y_val,y_pred=y_predict)
print(MSE**0.5)




################# LASSO ####################
from sklearn.linear_model import Lasso
lasso = Lasso(random_state = SEED)

# 라쏘 회귀 모델 훈련
lasso = Lasso(alpha=0.0001).fit(X_train, y_train)
# val 예측
y_predict = lasso.predict(X_val)
# 평가
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_true=y_val,y_pred=y_predict)
print(MSE**0.5)



################## RIDGE ####################
from sklearn.linear_model import Ridge
ridge = Ridge(random_state = SEED)

# 릿지 회귀 모델 훈련
ridge = Ridge(alpha=0.0001).fit(X_train, y_train) 
# val 예측
y_predict = ridge.predict(X_val)
# 모델 평가
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_true=y_val,y_pred=y_predict)
print(MSE**0.5)




################### Elasticnet ###################
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha= 0.01, l1_ratio = 0.5, random_state = SEED)

# 엘라스틱넷 훈련
elastic_net.fit(X_train,y_train)
# val 예측
y_predict = elastic_net.predict(X_val)
# 모델 평가
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_true=y_val,y_pred=y_predict)
print(MSE**0.5)

