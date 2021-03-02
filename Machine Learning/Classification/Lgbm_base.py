######## multiclass(>2) ########
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# adjust param
lgbm_param = {
    "objective": "multiclassova",
    'n_estimators' : num,
    "boosting": "gbdt",
    "num_leaves": 40,
    "learning_rate": 0.008,
    "feature_fraction": 0.9,
    "reg_lambda": 2,
    "metric": "multiclass",
    "num_class" : 3,
    'seed' : seed_num,
}

# train/valid data split
X_train, X_vallid, y_train, y_valid = train_test_split(X, 
                                                    y,
                                                    test_size=0.4, 
                                                    shuffle=False,
                                                    random_state=seed)


#make dataset
dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_valid, y_valid)



# train model
model = lgb.train(lgbm_param , dtrain,  num_boost_round = num, verbose_eval = verbose, early_stopping_rounds = stop_count,
                       valid_sets=(dtrain, dvalid), valid_names=('train','valid'))


# feature importance
lgb.plot_importance(model, importance_type='gain', max_num_features = max_num_features)


# predict test data
test_predict = model.predict(test)






######## binary(2) ######## 
