from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time                     # 시간 알고싶을때
from sklearn.svm import LinearSVR
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings ('ignore')

seed = 777

datasets = fetch_california_housing()
X = datasets.data
y = datasets.target
n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=seed)


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
model = RandomizedSearchCV(xgb, parameters, cv=kfold, verbose=1
                           , n_jobs=22
                           , n_iter=20
                           
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
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)

y_pred_best = model.best_estimator_.predict(X_test)
print("최적튠 r2 : " , r2_score(y_test, y_pred_best))
# best_score :  0.975 
# model.score :  0.9333333333333333
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# r2 score :  0.8050928958570419
# 최적튠 r2 :  0.8050928958570419
# 걸린시간 :  15.44 초

# 최적의 파라미터 :  {'subsample': 1, 'reg_lambda': 0, 'reg_alpha': 0.1, 'n_estimators': 500, 'min_child_weight': 0, 'max_depth': 7
# , 'learning_rate': 0.01, 'gamma': 1, 'colsample_bytree': 1, 'colsample_bynode': 0.5, 'colsample_bylevel': 1}
# best_score :  0.822171622656923
# model.score :  0.8273945793319387
# r2 score :  0.8273945793319387
# 최적튠 r2 :  0.8273945793319387
# 걸린시간 :  6.51 초