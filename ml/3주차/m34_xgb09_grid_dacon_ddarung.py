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
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings ('ignore')

seed = 777

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
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=seed)

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
xgb = XGBRegressor(random_state=seed)
model = RandomizedSearchCV(xgb, parameters, cv=kfold
                           , verbose=1
                           , n_jobs=22
                           , n_iter=20
                     )



start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)

print('best_score : ', model.best_score_)
print('model.score : ', model.score(X_test, y_test))

y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)

y_pred_best = model.best_estimator_.predict(X_test)
print("최적튠 r2 : " , r2_score(y_test, y_pred_best))

print("걸린시간 : ", round(end_time - start_time, 2), "초")

def RMSE(y_test, y_predict) :
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    return rmse

rmse = RMSE(y_test, y_predict)
print("rmse : ", rmse)

y_submit = model.predict(test_csv)
submission_csv['count'] = y_submit

submission_csv.to_csv(path + str(round(rmse, 3)) + ".csv", index=False)


# 최적의 파라미터 :  {'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0, 'n_estimators': 500, 'min_child_weight': 0.1, 'max_depth': 6
# , 'learning_rate': 0.1, 'gamma': 2, 'colsample_bytree': 1, 'colsample_bynode': 0.7, 'colsample_bylevel': 1}
# best_score :  0.783459730861022
# model.score :  0.7545349292921728
# r2 score :  0.7545349292921728
# 최적튠 r2 :  0.7545349292921728
# 걸린시간 :  2.88 초
# rmse :  38.1520340035067


