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
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
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
search_space = {
    'learning_rate' : hp.uniform('learning_rate', 0.001, 0.1),
    'max_depth' : hp.quniform('max_depth', 3, 10, 1.0),
    'num_leaves' : hp.quniform('num_leaves', 24, 40, 1.0),
    'min_child_samples' : hp.quniform('min_child_samples', 10, 200, 1.0),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 50, 1.0),
    'subsample' : hp.uniform('subsample', 0.5, 1),
    'colsample_bytree' : hp.quniform('colsample_bytree',0.5, 1, 1.0),
    'max_bin' :  hp.quniform('max_bin', 9, 500, 1.0),
    'reg_lambda' :  hp.quniform('reg_lambda', -0.001, 10, 1.0),
    'reg_alpha' :  hp.quniform('reg_alpha', 0.01, 50, 1.0),
}

def xgb_hamsu(search_space):
    
    params = {
        'n_estimators' : 100,
        'learning_rate' : search_space['learning_rate'],
        'max_depth' : int(search_space['max_depth']),
        'num_leaves' : int(search_space['num_leaves']),
        'min_child_samples' : int(search_space['min_child_samples']),
        'min_child_weight' : int(search_space['min_child_weight']),
        'subsample' : max(min(search_space['subsample'], 1), 0),
        'colsample_bytree' : search_space['colsample_bytree'],
        'max_bin' : max(int(search_space['max_bin']), 10),
        'reg_lambda' : max(search_space['reg_lambda'], 0),
        'reg_alpha' : search_space['reg_alpha'],
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

trial_val = Trials()
st = time.time()

best = fmin(
    fn = xgb_hamsu,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trial_val,
    rstate=np.random.default_rng(seed=10)
)
et = time.time()


# bay = BayesianOptimization(
#     f=xgb_hamsu,
#     pbounds=bayesian_params,
#     random_state=777
# )

n_iter = 100
# bay.maximize(init_points=5, n_iter=n_iter)
    
# print(bay.max)
print("best : ", best)
print(n_iter, "번 걸린시간 : ", round(et - st, 2), "초")\
    
# best :  {'colsample_bytree': 1.0, 'learning_rate': 0.003743385343536733, 'max_bin': 332.0, 'max_depth': 5.0, 'min_child_samples': 90.0, 'min_child_weight': 32.0, 'num_leaves': 29.0, 'reg_alpha': 43.0, 'reg_lambda': 4.0, 'subsample': 0.9299459818100199}
# 100 번 걸린시간 :  2.0 초

best_params = {
    'n_estimators': 300,
    'learning_rate': best['learning_rate'],
    'max_depth': int(best['max_depth']),
    'num_leaves': int(best['num_leaves']),
    'min_child_samples': int(best['min_child_samples']),
    'min_child_weight': int(best['min_child_weight']),
    'subsample': max(min(best['subsample'], 1), 0),
    'colsample_bytree': best['colsample_bytree'],
    'max_bin': max(int(best['max_bin']), 10),
    'reg_lambda': max(best['reg_lambda'], 0),
    'reg_alpha': best['reg_alpha'],
}

print(best_params)

model = XGBClassifier(**best_params, n_jobs=-1)
st = time.time()
model.fit(x_train, y_train)
et = time.time()

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("테스트 세트 정확도:", accuracy)
print("실행 시간:", round(et - st, 2), "초")

# 테스트 세트 정확도: 0.631578947368421
# 실행 시간: 0.01 초