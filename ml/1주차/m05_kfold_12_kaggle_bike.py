# https://www.kaggle.com/competitions/bike-sharing-demand/leaderboard
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import random
import time
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

#1. 데이터
path = "C://_data//kaggle//bike//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")

x = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']
test_csv = test_csv.drop([], axis=1)


n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2           
model = RandomForestRegressor()

#3
scores = cross_val_score(model, x, y, cv=kfold)  #cv 교차검증

print(f'r2 : {scores}\n평균 r2: {round(np.mean(scores), 4)}')

# r2 : [0.27321474 0.30163306 0.28476843 0.33252183 0.31884479]
# 평균 r2: 0.3022