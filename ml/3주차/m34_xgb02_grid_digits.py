from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

seed = 777

datasets = load_digits()

x = datasets.data
y = datasets.target
# print(x)
# print(y)
print(x.shape, y.shape) #(1797, 64) (1797,)

# print(pd.value_counts(y
#                       , sort=False
#                       ))
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

#  모델 : RF
# 그리드서치와 랜덤서치로 성능 및 시간 비교
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=seed, stratify=y)

# parameters = {
#     'n_estimators' : [100, 200, 300]
#     , 'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1]
#     , 'max_depth' : [2, 3, 5, 7, 8]
#     , 'gamma' : [0, 1, 2]
# }

parameters = {
    'n_estimators' : [100, 300, 500],
    'learning_rate' : [0.01, 0.1, 0.5],
    'max_depth' : [3, 4, 5, 6, 7, 8],
    'gamma' : [0, 1, 2, 3],
    'min_child_weight' : [0, 0.1, 0.5, 1],
    'subsample' : [0.5, 0.7, 1],
    'colsample_bytree' : [0.5, 0.7, 1],
    'colsample_bylevel' : [0.5, 0.7, 1],
    'colsample_bynode' : [0.5, 0.7, 1],
    'reg_alpha' : [0, 0.1, 0.5, 1],
    'reg_lambda' : [0, 0.1, 0.5, 1]
}

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

#2 모델
xgb = XGBClassifier(random_state=seed)
model = RandomizedSearchCV(xgb
                     , parameters
                     , cv=kfold
                     , verbose=1
                     , refit=True
                     , n_jobs=22
                     , random_state=seed 
                     , n_iter=20
                     )

import time

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

#4 평가, 예측
print(f"최상의 매개변수 : {model.best_estimator_}")
print(f"최상의 매개변수 : {model.best_params_}")
print(f"최상의 훈련 점수 : {model.best_score_}")

results = model.score(x_test, y_test)
print(f"최종 점수 : {results}")

# 그리드서치
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=10)
# 최적의 파라미터 :  {'min_samples_split': 3, 'n_jobs': 10}
# best_score :  0.975646051103368
# model.score :  0.9638888888888889
# accuracy_score :  0.9638888888888889
# 최적튠 ACC :  0.9638888888888889
# 걸린시간 :  4.47 초

#------------------------------------------------------------------------------#

# 랜덤서치
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=20)
# 최적의 파라미터 :  {'n_jobs': 20, 'min_samples_split': 3}
# best_score :  0.9749564459930313
# model.score :  0.9722222222222222
# accuracy_score :  0.9722222222222222
# 최적튠 ACC :  0.9722222222222222
# 걸린시간 :  1.95 초

# 

# 최적의 파라미터 :  {'subsample': 0.7, 'reg_lambda': 0, 'reg_alpha': 0, 'n_estimators': 500, 'min_child_weight': 0.5, 'max_depth': 8, 'learning_rate': 0.1, 'gamma': 0, 'colsample_bytree': 1, 'colsample_bynode': 1, 'colsample_bylevel': 0.5}
# best_score :  0.9714769647696478
# model.score :  0.9805555555555555
# accuracy_score :  0.9805555555555555
# 최적튠 ACC :  0.9805555555555555
# 걸린시간 :  6.85 초

# 최상의 매개변수 : {'subsample': 0.7, 'reg_lambda': 0, 'reg_alpha': 0, 'n_estimators': 500, 'min_child_weight': 0.5, 'max_depth': 8, 'learning_rate': 0.1, 'gamma': 0, 'colsample_bytree': 1, 'colsample_bynode': 1, 'colsample_bylevel': 0.5}
# 최상의 훈련 점수 : 0.9714769647696478
# 최종 점수 : 0.9805555555555555