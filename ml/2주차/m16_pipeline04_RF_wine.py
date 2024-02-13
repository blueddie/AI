from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras. callbacks import EarlyStopping
from sklearn.svm import LinearSVC
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score,StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import make_pipeline
warnings.filterwarnings ('ignore')


X, y = load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, test_size=0.2, stratify=y)

print(X_train.shape)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)



parameters = [
    {'n_estimators': [100,200], 'max_depth': [6,10,12],
     'min_samples_leaf' : [3, 10]},
    {'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_leaf' : [3, 5, 7, 10],
     'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5,10]},
    {'n_jobs' : [-1, 10, 20], 'min_samples_split' : [2, 3, 5, 10]}   
]

 #2. 모델 구성
model = make_pipeline(MinMaxScaler(), RandomForestClassifier(min_samples_split=3
                                                             , n_jobs=20
                                                             ))
  
#3 훈련
model.fit(X_train, y_train)

#4 평가
results = model.score(X_test, y_test)
print('model : ', " acc :", results)

# best_score :  0.9790640394088671
# model.score :  0.9722222222222222
# accuracy_score :  0.9722222222222222
# 최적튠 ACC :  0.9722222222222222
# 걸린시간 :  2.85 초
#--------------------------------------------------------------
# 랜덤 서치
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_split': 3}
# best_score :  0.9790640394088671
# model.score :  0.9722222222222222
# accuracy_score :  0.9722222222222222
# 최적튠 ACC :  0.9722222222222222
# 걸린시간 :  1.56 초

# Fitting 5 folds for each of 20 candidates, totalling 100 fits
# 최적의 매개변수 :  RandomForestClassifier(n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_split': 2}
# best_score :  0.9790640394088671
# model.score :  1.0
# accuracy_score :  1.0
# 최적튠 ACC :  1.0
# 걸린시간 :  1.86 초

#=============================================================================
# (142, 13)
# n_iterations: 3
# n_required_iterations: 4
# n_possible_iterations: 3
# min_resources_: 10
# max_resources_: 142
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 60
# n_resources: 10
# Fitting 5 folds for each of 60 candidates, totalling 300 fits
# ----------
# iter: 1
# n_candidates: 20
# n_resources: 30
# Fitting 5 folds for each of 20 candidates, totalling 100 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 90
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=20)
# 최적의 파라미터 :  {'min_samples_split': 3, 'n_jobs': 20}
# best_score :  0.977124183006536
# model.score :  1.0
# accuracy_score :  1.0
# 최적튠 ACC :  1.0
# 걸린시간 :  3.71 초

# model :   acc : 1.0