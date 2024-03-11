import numpy as np
import tensorflow as tf
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
import time
import warnings
np.random.seed(777)
tf.random.set_seed(777)
random.seed(777)

warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2.모델
bayesian_params = {
    'learning_rate' : (0.001, 0.5),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (9,500),
    'reg_lambda' : (-0.001, 10),
    'reg_alpha' : (0.01, 50),
}

def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample, 1), 0),
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)), 10),
        'reg_lambda' : max(reg_lambda, 0),
        'reg_alpha' : reg_alpha,
    }
    
    model = XGBClassifier()
    model.set_params(**params, n_jobs=-1)
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='logloss',
              early_stopping_rounds=50,
              verbose=0
              )
    y_pred = model.predict(x_test)
    results = accuracy_score(y_test, y_pred)
    
    return results

bay = BayesianOptimization(
    f=xgb_hamsu,
    pbounds=bayesian_params,
    random_state=777
)

n_iter = 100
st = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
et = time.time()
    
print(bay.max)
print(n_iter, "번 걸린시간 : ", round(et - st, 2), "초")\
    
# {'target': 0.9824561403508771, 'params': {'colsample_bytree': 0.5, 'learning_rate': 1.0, 'max_bin': 429.6353844524899, 'max_depth': 3.0, 'min_child_samples': 73.18196528806162, 'min_child_weight': 6.27601799782907, 'num_leaves': 40.0, 'reg_alpha': 0.01, 'reg_lambda': -0.001, 'subsample': 1.0}}    
# 100 번 걸린시간 :  22.32 초

# {'target': 0.9912280701754386, 'params': {'colsample_bytree': 0.5, 'learning_rate': 0.5, 'max_bin': 212.17316244783004, 'max_depth': 3.0, 'min_child_samples': 200.0, 'min_child_weight': 1.0, 'num_leaves': 24.0, 'reg_alpha': 0.01, 'reg_lambda': 10.0, 'subsample': 1.0}}
# 100 번 걸린시간 :  24.35 초