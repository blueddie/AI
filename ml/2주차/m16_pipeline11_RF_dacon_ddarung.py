# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import LinearSVR
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

warnings.filterwarnings ('ignore')


# 1.데이터

path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)        # 컬럼 지정         , # index_col = : 지정 안해주면 인덱스도 컬럼 판단

test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "submission.csv")

train_csv['hour_bef_precipitation'] = train_csv['hour_bef_precipitation'].fillna(0)
train_csv['hour_bef_pm10'] = train_csv['hour_bef_pm10'].fillna(0)
train_csv['hour_bef_pm2.5'] = train_csv['hour_bef_pm2.5'].fillna(0)
train_csv['hour_bef_windspeed'] = train_csv['hour_bef_windspeed'].fillna(0)
train_csv['hour_bef_temperature'] = train_csv['hour_bef_temperature'].fillna(train_csv['hour_bef_temperature'].mean())
train_csv['hour_bef_humidity'] = train_csv['hour_bef_humidity'].fillna(train_csv['hour_bef_humidity'].mean())
train_csv['hour_bef_visibility'] = train_csv['hour_bef_visibility'].fillna(train_csv['hour_bef_visibility'].mean())
train_csv['hour_bef_ozone'] = train_csv['hour_bef_ozone'].fillna(train_csv['hour_bef_ozone'].mean())

test_csv = test_csv.fillna(test_csv.mean())
# print(test_csv.info())      # 717 non-null

X = train_csv.drop(['count'], axis=1)
y = train_csv['count']
n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=713)

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
model = HalvingGridSearchCV(RandomForestRegressor()
                            , parameters
                            , cv=kfold
                            , verbose=1
                            , factor=3
                    # refit = True,     # default
                    #  n_jobs=-1
                     )



start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  SVC(C=1, kernel='linear')

print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}

print('best_score : ', model.best_score_)
print('model.score : ', model.score(X_test, y_test))
# results = model.score(X_test, y_test)
# print(results)
y_predict = model.predict(X_test)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)

y_pred_best = model.best_estimator_.predict(X_test)
# print("최적튠 ACC : " , accuracy_score(y_test, y_pred_best))
# best_score :  0.975 
# model.score :  0.9333333333333333
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# best_score :  0.7791705364063007
# model.score :  0.8707761578141365

# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 48
# max_resources_: 1313
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 60
# n_resources: 48
# Fitting 5 folds for each of 60 candidates, totalling 300 fits
# ----------
# iter: 1
# n_candidates: 20
# n_resources: 144
# Fitting 5 folds for each of 20 candidates, totalling 100 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 432
# Fitting 5 folds for each of 7 candidates, totalling 35 fits
# ----------
# iter: 3
# n_candidates: 3
# n_resources: 1296
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestRegressor(n_jobs=20)
# 최적의 파라미터 :  {'min_samples_split': 2, 'n_jobs': 20}
# best_score :  0.7703320183451801
# model.score :  0.8762464912580052
# 걸린시간 :  46.25 초