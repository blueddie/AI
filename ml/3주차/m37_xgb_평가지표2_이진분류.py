from sklearn.datasets import load_diabetes, load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score, roc_auc_score
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
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings ('ignore')

seed = 777

X, y = load_breast_cancer(return_X_y=True)

n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.2, stratify=y)

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
model = XGBClassifier(random_state=seed)

# set_param4
model.set_params(
                 early_stopping_rounds=10
                 , **parameters
                 )


#3 훈련
start_time = time.time()

model.fit(X_train, y_train
          , eval_set=[(X_train, y_train), (X_test, y_test)]
          , verbose=1
        #   , eval_metric='rmse'        # 회귀 디폴트
        #   , eval_metric='mae'         # rmsle, mape, mphe.. 등등...

          # , eval_metric='logloss'     # 이진 분류 디폴트  ACC
        #   , eval_metric='mlogloss'      # 다중 분류 디폴트 ACC
          
        #   , eval_metric='error'       # 이진 분류
        #   , eval_metric='merror'    # 다중 분류
          # , eval_metric='auc'     # 이진 분류, 다중 분류 (하지만 이진 분류에 좋다)
          )

end_time = time.time()

# print(f"파라미터 : {model.get_params()}")
print('최종 점수 : ', model.score(X_test, y_test))

print("걸린시간 : ", round(end_time - start_time, 2), "초")

y_predict = model.predict(X_test)

acc = accuracy_score(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average='macro')
roc_auc = roc_auc_score(y_test, y_predict)

print(f"acc : {acc}")
print(f"r2 : {r2}")
print(f"roc_auc : {roc_auc}")
print(f"f1 : {f1}")