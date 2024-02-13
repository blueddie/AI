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
warnings.filterwarnings ('ignore')


datasets= load_breast_cancer()

X = datasets.data
y = datasets.target


n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False, random_state=3)

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

 #2. 모델 구성
model = RandomizedSearchCV(RandomForestClassifier()
                           , parameters
                           , cv=kfold
                           , verbose=1
                           , n_jobs=-1
                           , n_iter=30
                           , random_state=42
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
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

y_pred_best = model.best_estimator_.predict(X_test)
print("최적튠 ACC : " , accuracy_score(y_test, y_pred_best))
# best_score :  0.975 
# model.score :  0.9333333333333333
print("걸린시간 : ", round(end_time - start_time, 2), "초")


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