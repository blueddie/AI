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
import warnings
import numpy as np
import time
from sklearn.metrics import r2_score

warnings.filterwarnings ('ignore')

X, y = load_diabetes(return_X_y=True)

n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, test_size=0.2)



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
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1,
                    # refit = True,     # default
                     n_jobs=-1)



start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print("최적의 매개변수 : ", model.best_estimator_)

print("최적의 파라미터 : ", model.best_params_)

print('best_score : ', model.best_score_)
print('model.score : ', model.score(X_test, y_test))

y_predict = model.predict(X_test)

y_pred_best = model.best_estimator_.predict(X_test)
r2 = r2_score(y_test, y_predict)
print(f"r2 score: {r2}")
print("걸린시간 : ", round(end_time - start_time, 2), "초")



# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, min_samples_leaf=10)
# 최적의 파라미터 :  {'min_samples_leaf': 10, 'max_depth': 10}
# best_score :  0.46030451194910427
# model.score :  0.41341699450017166
# r2 score: 0.41341699450017166
# 걸린시간 :  1.57 초초

