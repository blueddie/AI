import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import time

#1. 데이터
# x, y = load_iris(return_X_y=True)
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=123, train_size=0.8, stratify=y)

print(x_train.shape)    #(1437, 64)

scaler =  StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    
parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3,4,5]}
    , {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]}
    , {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.01, 0.001, 0.0001], "degree":[3, 4]}
         
]

#2 모델
# model = SVC(C=1, kernel='linear', degree=3)
# model = GridSearchCV(SVC()
#                      , parameters
#                      , cv=kfold
#                      , verbose=1
#                      , refit=True   # 가장 좋았던 걸 다시 훈련시키겠다. 디폴트 True
#                      , n_jobs=-1  
#          
print("===============================하빙 그리드 시작=====================")
model = HalvingGridSearchCV(SVC()
                     , parameters
                     , cv=5
                     , verbose=1
                     , refit=True   # 가장 좋았던 걸 다시 훈련시키겠다. 디폴트 True
                     , n_jobs=-1  
                     , random_state=66
                     , factor=3 # 디폴트 3
                     , min_resources=150
                    #  , n_iter=20    # 디폴트 10
                     )

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()
#3
print(f"최적의 매개변수 : {model.best_estimator_}\n최적의 파라미터 : {model.best_params_}")
print(f"best score : {model.best_score_}\nmodel.score : {model.score(x_test, y_test)}")

# 최적의 매개변수 : SVC(C=1, kernel='linear')                      # 
# 최적의 파라미터 : {'C': 1, 'degree': 3, 'kernel': 'linear'}      # 
# best score : 0.975                                              #
# model.score : 0.9666666666666667                                #


y_predict = model.predict(x_test)
print("ACC : ", accuracy_score(y_test, y_predict))

y_predict_best = model.best_estimator_.predict(x_test)
            # SVC(C=1, kernel='linear').predict(x_test)
print("최적튠 ACC : ", accuracy_score(y_test, y_predict_best))
print(f"걸린 시간 : {round(end_time - start_time, 2)} 초")

import pandas as pd
# print(pd.DataFrame(model.cv_results_).T)

import sklearn as sk
print(sk.__version__)   #1.1.3

# ===============================하빙 그리드 시작=====================
# n_iterations: 2
# n_required_iterations: 4
# n_possible_iterations: 2
# min_resources_: 30                        
# max_resources_: 120
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 42
# n_resources: 30
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# ----------
# iter: 1
# n_candidates: 14
# n_resources: 90
# Fitting 5 folds for each of 14 candidates, totalling 70 fits
# 최적의 매개변수 : SVC(C=1, degree=4, kernel='linear')
# 최적의 파라미터 : {'C': 1, 'degree': 4, 'kernel': 'linear'}
# best score : 0.9777777777777779
# model.score : 0.9
# ACC :  0.9
# 최적튠 ACC :  0.9
# 걸린 시간 : 1.21 초
# 1.1.3


# ===============================하빙 그리드 시작=====================
# n_iterations: 2
# n_required_iterations: 3
# n_possible_iterations: 2
# min_resources_: 100                           # cv * 2 * 라벨의 갯수
# max_resources_: 1437
# aggressive_elimination: False
# factor: 4
# ----------
# iter: 0
# n_candidates: 42
# n_resources: 100
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# ----------
# iter: 1
# n_candidates: 11
# n_resources: 400
# Fitting 5 folds for each of 11 candidates, totalling 55 fits
# 최적의 매개변수 : SVC(C=1000, degree=4, gamma=0.0001, kernel='sigmoid')
# 최적의 파라미터 : {'C': 1000, 'degree': 4, 'gamma': 0.0001, 'kernel': 'sigmoid'}
# best score : 0.9445886075949368
# model.score : 0.9888888888888889
# ACC :  0.9888888888888889
# 최적튠 ACC :  0.9888888888888889
# 걸린 시간 : 1.46 초
# 1.1.3