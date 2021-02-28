#need to install

#pycaret autoML(classification)
from pycaret.classification import *

#분석 데이터 설정
clf = setup(data = train, target = Target_var , session_id = SEED , silent = True)

#모델 비교
best_3 = compare_models(sort = 'F1', n_select = 3)

#모델 앙상블
blended = blend_models(estimator_list = best_3, fold = 5, method = 'soft')

#모델 평가
pred_holdout = predict_model(blended)

#전체 데이터 학습
final_model = finalize_model(blended)

#테스트 데이터 예측
predictions = predict_model(final_model, data = test)

