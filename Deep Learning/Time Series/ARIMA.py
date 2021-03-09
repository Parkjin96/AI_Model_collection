#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from pylab import rcParams
import statsmodels.api as sm
import warnings
import itertools
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA

# Kaggle에서 데이터를 받아오면 train.csv와 test.csv가 있는데 train.csv를 
df = pd.read_csv("data/train.csv")

# string 형태의 date 필드를 datetime 형태로 변환
df['date'] =  pd.to_datetime(df['date'])

# date 필드를 index로 설정
df = df.set_index('date')

# 빠르게 테스트 해 보기 위해 월별로 아이템 판매 예측을 해 보기로 함
salesbymonth = df.sales.resample('M').sum()

#2013-2016 데이터를 train으로 2017 데이터를 test로 분리
split = "2017-01-01"
salesbymonth_train= salesbymonth[:split]
salesbymonth_test= salesbymonth[split:]
salesbymonth_test_final=salesbymonth_test.copy()

# 데이터를 시즌별로 분해해서 살펴 봄
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(salesbymonth_train, model='additive')
fig = decomposition.plot()
plt.show()


# In[ ]:


#방법 1. p,d,q의 조합을 만들어 하나하나 ARIMA 모델을 돌려봄
p = d = q = range(0, 2)

import itertools
pdqa = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

for param in pdqa:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(salesbymonth_train, order=param, seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)                                
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
            
#방법 2. auto_arima 함수로 자동 추출
from pmdarima import auto_arima
stepwise_model = auto_arima(salesbymonth_train, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)


# In[ ]:


SARIMAMonth = sm.tsa.statespace.SARIMAX(salesbymonth, order=(0, 1, 0), seasonal_order=(1, 1, 0, 12) ,enforce_stationarity=False,enforce_invertibility=False)

SARIMA_results_month = SARIMAMonth.fit()

SARIMA_results_month.plot_diagnostics(figsize=(16, 8))
plt.show()


# In[ ]:


# 2017년 12개월 데이터로 예측
SARIMA_predict_month_1 = SARIMA_results_month.predict(start=48,end=60)

# 결과 비교를 위해 기존에 마련해둔 test데이터에 결과를 붙임
salesbymonth_test_final['SeasonalARIMA'] = SARIMA_predict_month_1

# RMSE를 살펴 봄
RMSE_Month_Seasonal_ARIMA  = np.mean(np.sqrt((salesbymonth_test_final['SeasonalARIMA'] - salesbymonth_test_final['sales']) ** 2)) 
print(RMSE_Month_Seasonal_ARIMA)
-> 12190.886296802802

# test 데이터와 예측 결과치를 비교
salesbymonth_test_final[1:].plot()

