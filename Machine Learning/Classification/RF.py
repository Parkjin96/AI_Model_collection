#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#### RandomForest ####
from sklearn.ensemble import RandomForestClassifier

# 파라미터 설정
rf = RandomForestClassifier(n_estimators=50,
                            random_state = SEED,
                            max_depth=40, 
                            max_leaf_nodes=20,
                            n_jobs=-1
                            )
            
# 모델 학습
model = rf.fit(X_train, y_train)

# 모델 평가
predictions = model.predict(X_val)
proba = model.predict_proba(X_val)

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score, recall_score
score = roc_auc_score(y_val, predictions)
score2 = accuracy_score(y_val, predictions)
score3 = f1_score(y_val, predictions)
score4 = recall_score(y_val, predictions)

