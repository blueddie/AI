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

# parameters = {
#     'n_estimators' : [100, 300, 500],
#     'learning_rate' : [0.01, 0.1, 0.5],
#     'max_depth' : [3, 4, 5, 6, 7, 8],
#     'gamma' : [0, 1, 2, 3],
#     'min_child_weight' : [0, 0.1, 0.5, 1],
#     'subsample' : [0.5, 0.7, 1],
#     'colsample_bytree' : [0.5, 0.7, 1],
#     'colsample_bylevel' : [0.5, 0.7, 1],
#     'colsample_bynode' : [0.5, 0.7, 1],
#     'reg_alpha' : [0, 0.1, 0.5, 1],
#     'reg_lambda' : [0, 0.1, 0.5, 1]
# }

parameters = {
    'n_estimators' : 100
    , 'learning_rate' : 0.1
    , 'max_depth' : 6
    , 'min_child_weight' : 10
}


 #2. 모델 구성
# model = XGBRegressor(random_state=seed, **parameters)
model = XGBRegressor(random_state=seed)

# set_param4
model.set_params(
                 early_stopping_rounds=10
                 , **parameters
                 )

start_time = time.time()

model.fit(X_train, y_train
          , eval_set=[(X_train, y_train), (X_test, y_test)]
          , verbose=1
          )

end_time = time.time()

# print(f"파라미터 : {model.get_params()}")
print('최종 점수 : ', model.score(X_test, y_test))

print("걸린시간 : ", round(end_time - start_time, 2), "초")

