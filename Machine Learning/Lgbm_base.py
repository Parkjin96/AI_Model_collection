#!/usr/bin/env python
# coding: utf-8

# In[3]:


import lightgbm as lgb
from sklearn.model_selection import train_test_split


# In[2]:


# adjust param
lgbm_param = {
    "objective": "multiclassova",
    'n_estimators' : NUM_BOOST_ROUND,
    "boosting": "gbdt",
    "num_leaves": 40,
    "learning_rate": 0.008,
    "feature_fraction": 0.9,
    "reg_lambda": 2,
    "metric": "multiclass",
    "num_class" : 3,
    'seed' : SEED,
}


# In[ ]:


# train/valid data split
X_train, X_vallid, y_train, y_valid = train_test_split(X, 
                                                    y,
                                                    test_size=0.4, 
                                                    shuffle=False,
                                                    random_state=seed)


# In[ ]:


#make dataset
dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_valid, y_valid)


# In[ ]:


# train model
model = lgb.train(lgbm_param , dtrain,  num_boost_round = num, verbose_eval = verbose, early_stopping_rounds = stop_count,
                       valid_sets=(dtrain, dvalid), valid_names=('train','valid'))


# In[ ]:


# feature importance
lgb.plot_importance(model, importance_type='gain', max_num_features = max_num_features)


# In[ ]:


# predict test data
test_predict = model.predict(test)

