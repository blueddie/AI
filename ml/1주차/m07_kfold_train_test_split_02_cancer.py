import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import time
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=123, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2           
model = AdaBoostClassifier()
#3
scores = cross_val_score(model, x_train, y_train, cv=kfold)  #cv 교차검증

print(f'ACC : {scores}\n평균 ACC: {round(np.mean(scores), 4)}')

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
# print(y_predict)
# print(y_test)
acc = accuracy_score(y_test, y_predict)
print(f'cross_val_predict ACC : {acc} ')
# ACC : [0.97368421 0.97368421 0.95614035 0.94736842 0.9380531 ]
# 평균 ACC: 0.9578

# ACC : [0.94736842 0.93859649 1.         0.98245614 0.98230088]
# 평균 ACC: 0.9701

# 평균 ACC: 0.9604
# cross_val_predict ACC : 0.9473684210526315 