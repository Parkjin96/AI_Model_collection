#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1. 함수 홀출 및 환경 설정
#함수 호출
from pycaret.regression import *
#환경 설정
reg = setup(train, target = 'SalePrice', train_size=0.8)



# 2. 모델 비교
best = compare_models(sort = 'RMSE') # sort에 평가 지표설정(ex : f1, auc, recall...)



# 3. 모델 생성 ( 모델 비교 결과 참고 )
cat = create_model('catboost', cross_validation = False)
xgb = create_model('xgboost', cross_validation = False)
gbr = create_model('gbr', cross_validation = False)



# 4. 하이퍼파라미터 튜닝
"""
모델 튜닝에는 기본적으로 K-Fold CV를 바탕으로 진행되며 최적화는 과업에 맞게 'RMSE"를 선택해 진행했다.

튜닝 방법은 일반적으로 Random Grid를 iter(default=10)만큰 최적화 시킨다.

해당 방법 외 custom grid 파라미터를 추가하여 튜닝을 진행할 수 있다.
"""
tuned_cat = tune_model(cat, optimize = 'RMSE', n_iter = 10)
tuned_xgb = tune_model(xgb, optimize = 'RMSE', n_iter = 10)
tuned_gbr = tune_model(gbr, optimize = 'RMSE', n_iter = 10)




# 5. 모델 블랜딩
blender_specific = blend_models(estimator_list = [tuned_cat,tuned_xgb,tuned_gbr], optimize = 'RMSE')





# 6. 시각화
# 모델 시각화(plot)_plot = 'residuals'
plot_model(blender_specific)
# 모델 시각화(plot)_plot = 'error'
plot_model(blender_specific, plot='error')
# 모델 시각화(plot)_plot = 'learning'
plot_model(blender_specific, plot='learning')



# 7. 전체 데이터 학습 및 예측
# 마지막 학습(Finalize)
final_model = finalize_model(blender_specific)
# 예측(Predict)
pred = predict_model(final_model, data = test)

