# https://www.kaggle.com/competitions/bike-sharing-demand/leaderboard
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.linear_model import LinearRegression
import random
import time

#1. 데이터
path = "C://_data//kaggle//bike//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)  [10886 rows x 11 columns]
test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv)   [6493 rows x 8 columns]
submission_csv = pd.read_csv(path + "sampleSubmission.csv")
# print(submission_csv) [6493 rows x 2 columns]


X = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']
# print(X.shape)   #(10886, 8)
# print(y.shape)  #(10886,)
# print(test_csv.shape) #(6493, 8)

#2. 모델
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

def RMSE(y_test, y_predict):
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    
    return rmse

def auto(a,b) :
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=a)
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=b, validation_split=0.2)
    loss = model.evaluate(X_test, y_test)
    y_predict = model.predict(X_test)
    r2 = r2_score(y_test, y_predict)
    y_submit = model.predict([test_csv])
    submission_csv['count'] = y_submit
    if submission_csv['count'].min() > 0 :
        
        rmse = RMSE(y_test, y_predict)
        return rmse, loss, r2
  
  
min_rmse = 1600
max_r2 = 0.2  
date = "0109_"
last_name = ".csv"
i = 1

while True :
    # batch_size = random.randrange(256, 513)
    # random_state = random.randrange(1, 999999)
    rmse, loss, r2 = auto(7, 150)
    if  rmse < min_rmse and r2 > max_r2:
        min_rmse = rmse
        max_r2 = r2
        rmse = round(rmse, 2)
        submission_csv.to_csv(path + date + str(i) +"andRMSE_" + last_name , index=False)
        i += 1
        
        리스트 = [1,2,3,4,5]
        튜플 = (1,2,3,4,5)
        딕셔너리 = {'과일' : '사과','과일' : '사과', '가격' : '1200', '생산지' : '문경'}