from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from sklearn.svm import LinearSVC
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import time
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings ('ignore')

seed = 777

X, y = load_diabetes(return_X_y=True)

n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.2)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

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

 #2. 모델 구성
model = XGBRegressor(random_state=seed)



start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
# print("최적의 매개변수 : ", model.best_estimator_)
# print("최적의 파라미터 : ", model.best_params_)

# print('best_score : ', model.best_score_)
print('최종 점수 : ', model.score(X_test, y_test))

y_predict = model.predict(X_test)

r2 = r2_score(y_test, y_predict)
print(f"r2 score: {r2}")
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# set_param2
model.set_params(gamma=0.3)

start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()

print('최종 점수2 : ', model.score(X_test, y_test))

y_predict = model.predict(X_test)

r2 = r2_score(y_test, y_predict)
print(f"r2 score2 : {r2}")
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# set_param3
model.set_params(learing_rate=0.01)

start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()

print('최종 점수3 : ', model.score(X_test, y_test))

y_predict = model.predict(X_test)

r2 = r2_score(y_test, y_predict)
print(f"r2 score3 : {r2}")
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# set_param4
model.set_params(learing_rate=0.01
                 , max_depth=3
                 , reg_lambda=1
                 , reg_alpha=0
                #  , min_child
                 )

start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()

print('최종 점수4 : ', model.score(X_test, y_test))

y_predict = model.predict(X_test)

r2 = r2_score(y_test, y_predict)
print(f"r2 score4 : {r2}")
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# 최종 점수 :  0.19902639314080917
# r2 score: 0.19902639314080917
# 걸린시간 :  0.05 초
# 최종 점수2 :  0.21239130913036297
# r2 score2 : 0.21239130913036297
# 걸린시간 :  0.02 초

