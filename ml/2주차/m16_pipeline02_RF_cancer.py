import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.svm import LinearSVC
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
import time
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import make_pipeline

warnings.filterwarnings ('ignore')


datasets= load_breast_cancer()

X = datasets.data
y = datasets.target


n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False, random_state=3)

print(X_train.shape)
mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)




parameters = [
    {'n_estimators': [100,200], 'max_depth': [6,10,12],
     'min_samples_leaf' : [3, 10]},
    {'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_leaf' : [3, 5, 7, 10],
     'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5,10]},
    {'n_jobs' : [-1, 10, 20], 'min_samples_split' : [2, 3, 5, 10]}   
]


#2 모델
# model = RandomForestClassifier()
model = make_pipeline(MinMaxScaler(), RandomForestClassifier(max_depth=12
                                                             , min_samples_leaf=3
                                                             , n_estimators=200
                                                             ))
  
#3 훈련
model.fit(X_train, y_train)

#4 평가
results = model.score(X_test, y_test)
print('model : ', " acc :", results)


# 최적의 매개변수 :  RandomForestClassifier(max_depth=12, min_samples_leaf=3)
# 최적의 파라미터 :  {'max_depth': 12, 'min_samples_leaf': 3}
# best_score :  0.9559249786871271
# model.score :  0.9517543859649122
# accuracy_score :  0.9517543859649122
# 최적튠 ACC :  0.9517543859649122
# 걸린시간 :  3.07 초


#--------------------------------------------
# 랜덤서치
# 최적의 매개변수 :  RandomForestClassifier(max_depth=12, min_samples_leaf=3)
# 최적의 파라미터 :  {'min_samples_leaf': 3, 'max_depth': 12}
# best_score :  0.9529838022165389
# model.score :  0.956140350877193
# accuracy_score :  0.956140350877193
# 최적튠 ACC :  0.956140350877193
# 걸린시간 :  1.82 초

# Fitting 5 folds for each of 20 candidates, totalling 100 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=5, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_split': 5}
# best_score :  0.9530264279624895
# model.score :  0.9605263157894737
# accuracy_score :  0.9605263157894737
# 최적튠 ACC :  0.9605263157894737
# 걸린시간 :  1.95 초


# (341, 30)
# n_iterations: 3
# n_required_iterations: 4
# n_possible_iterations: 3
# min_resources_: 20
# max_resources_: 341
# aggressive_elimination: False
# factor: 3.6
# ----------
# iter: 0
# n_candidates: 60
# n_resources: 20
# Fitting 5 folds for each of 60 candidates, totalling 300 fits
# ----------
# iter: 1
# n_candidates: 17
# n_resources: 72
# Fitting 5 folds for each of 17 candidates, totalling 85 fits
# ----------
# iter: 2
# n_candidates: 5
# n_resources: 259
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# 최적의 매개변수 :  RandomForestClassifier(max_depth=12, min_samples_leaf=3, n_estimators=200)
# 최적의 파라미터 :  {'max_depth': 12, 'min_samples_leaf': 3, 'n_estimators': 200}
# best_score :  0.9569381598793363
# model.score :  0.956140350877193
# accuracy_score :  0.956140350877193
# 최적튠 ACC :  0.95614035
##############################################################
# 파이프 라인
# model :   acc : 0.9605263157894737