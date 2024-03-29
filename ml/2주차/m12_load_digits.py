from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

datasets = load_digits()

x = datasets.data
y = datasets.target
# print(x)
# print(y)
print(x.shape, y.shape) #(1797, 64) (1797,)

# print(pd.value_counts(y
#                       , sort=False
#                       ))
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

#  모델 : RF
# 그리드서치와 랜덤서치로 성능 및 시간 비교
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42, stratify=y)

parameters = [
    {'n_estimators': [100,200], 'max_depth': [6,12,18],
     'min_samples_leaf' : [3, 10]},
    {'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_leaf' : [3, 5, 7, 10],
     'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5,10]},
    {'n_jobs' : [-1, 10, 20], 'min_samples_split' : [2, 3, 5, 10]}   
]

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2 모델
model = RandomizedSearchCV(RandomForestClassifier()
                     , parameters
                     , cv=kfold
                     , verbose=1
                     , refit=True
                     , n_jobs=-1  
                     )

import time

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)

print('best_score : ', model.best_score_)
print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

y_pred_best = model.best_estimator_.predict(x_test)
print("최적튠 ACC : " , accuracy_score(y_test, y_pred_best))

print("걸린시간 : ", round(end_time - start_time, 2), "초")

# 그리드서치
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=10)
# 최적의 파라미터 :  {'min_samples_split': 3, 'n_jobs': 10}
# best_score :  0.975646051103368
# model.score :  0.9638888888888889
# accuracy_score :  0.9638888888888889
# 최적튠 ACC :  0.9638888888888889
# 걸린시간 :  4.47 초

#------------------------------------------------------------------------------#

# 랜덤서치
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=20)
# 최적의 파라미터 :  {'n_jobs': 20, 'min_samples_split': 3}
# best_score :  0.9749564459930313
# model.score :  0.9722222222222222
# accuracy_score :  0.9722222222222222
# 최적튠 ACC :  0.9722222222222222
# 걸린시간 :  1.95 초