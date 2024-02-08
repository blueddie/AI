# https://dacon.io/competitions/open/235576/codeshare/6969?page=1&dtype=recent // 대회 주소
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
import numpy as np

#1. 데이터
path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path + "submission.csv")   #, index_col=0
print(submission_csv)

X = train_csv.drop(['count'], axis=1)

y = train_csv["count"]

test_csv = test_csv.fillna(test_csv.mean())

X['hour_bef_temperature'] = X['hour_bef_temperature'].fillna(X['hour_bef_temperature'].mean())

X.loc[[1420],['hour_bef_precipitation']] = 0.0
X.loc[[1553],['hour_bef_precipitation']] = 1.0

X.loc[[1420],['hour_bef_humidity']] = 37.0
X.loc[[1553],['hour_bef_humidity']] = 82.0

X['hour_bef_visibility'] = X['hour_bef_visibility'].fillna(X['hour_bef_visibility'].mean())

X = X.fillna(X.mean())
# first_name = "submission_0110_es_rs_"
second_name = ".csv"

print(X.shape)  #(1459, 9)


x = X.astype(np.float32)
test_csv = test_csv.astype(np.float32)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2           
model = RandomForestRegressor()

#3
scores = cross_val_score(model, x, y, cv=kfold)  #cv 교차검증

print(f'r2 : {scores}\n평균 r2: {round(np.mean(scores), 4)}')

# r2 : [0.79055066 0.76818674 0.76433662 0.78806186 0.78104707]
# 평균 r2: 0.7784