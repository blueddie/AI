from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import datetime
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
import numpy as np
#1
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.8)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2           
model = RandomForestRegressor()

#3
scores = cross_val_score(model, x, y, cv=kfold)  #cv 교차검증

print(f'r2 : {scores}\n평균 r2: {round(np.mean(scores), 4)}')

# r2 : [0.81329299 0.82471812 0.81016245 0.79435548 0.80404442]
# 평균 r2: 0.8093