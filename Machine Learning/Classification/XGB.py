#### XGB(Classifier) ####
import xgboost as xgb

# 반드시 튜닝해야할 파라미터는  min_child_weight / max_depth / gamma
xgb_model = xgb.XGBClassifier(
    booster='gbtree',      # 트리,회귀(gblinear) 트리가 항상 
                           # 더 좋은 성능을 내기 때문에 수정할 필요없다고한다.
    
    silent=False,          # running message출력안한다.
                           # 모델이 적합되는 과정을 이해하기위해선 False으로한다.
    
    min_child_weight=10,   # 값이 높아지면 under-fitting 되는 
                           # 경우가 있다. CV를 통해 튜닝되어야 한다.
    
    max_depth=8,           # 트리의 최대 깊이를 정의함. 
                           # 루트에서 가장 긴 노드의 거리.
                           # 8이면 중요변수에서 결론까지 변수가 9개거친다.
                           # Typical Value는 3-10. 
    
    gamma =0,              # 노드가 split 되기 위한 loss function의 값이
                           # 감소하는 최소값을 정의한다. gamma 값이 높아질 수록 
                           # 알고리즘은 보수적으로 변하고, loss function의 정의에따라 적정값이 달라지기때문에 반드시 튜닝.
    
    nthread =4,            # XGBoost를 실행하기 위한 병렬처리(쓰레드)
                           #갯수. 'n_jobs' 를 사용해라.
    
    colsample_bytree=0.8,  # 트리를 생성할때 훈련 데이터에서 
                           # 변수를 샘플링해주는 비율. 보통0.6~0.9
    
    colsample_bylevel=0.9, # 트리의 레벨별로 훈련 데이터의 
                           #변수를 샘플링해주는 비율. 보통0.6~0.9
     
    n_estimators =30,     # 트리의 갯수. 
    
    """
    objective = 'reg:linear','binary:logistic','multi:softmax',
                 'multi:softprob'  # 4가지 존재.
            # 회귀 경우 'reg', binary분류의 경우 'binary',
            # 다중분류경우 'multi'- 분류된 class를 return하는 경우 'softmax'
            # 각 class에 속할 확률을 return하는 경우 'softprob'
   """
    
    metrics ='f1',
    random_state =  SEED
)



#### 학습 ####
model = xgb_model.fit(X_train,y_train,eval_set = [(X_val,y_val)],eval_metric = 'auc',early_stopping_rounds=10,verbose=5)
#     X (array_like)     # Feature matrix ( 독립변수)
#                        # X_train
    
#     Y (array)          # Labels (종속변수)
#                        # Y_train
    
#     eval_set           # 빨리 끝나기 위해 검증데이터와 같이써야한다.  
#                        # =[(X_train,Y_train),(X_vld, Y_vld)]
 
#     eval_metric = 'rmse','error','mae','logloss','merror','mlogloss','auc'  
         
#     early_stopping_rounds=100,20 :100번,20번 반복동안 향상 되지 않으면 stop


#### Featurne imoprtance ####
plot_importance(model)
pyplot.show()



#### 평가 ####
predictions = model.predict(X_val)
proba = model.predict_proba(X_val)
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score, recall_score
score = roc_auc_score(y_val, predictions)
score2 = accuracy_score(y_val, predictions)
score3 = f1_score(y_val, predictions)
score4 = recall_score(y_val, predictions)

