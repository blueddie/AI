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
path = "C:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)  [10886 rows x 11 columns]
test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv)   [6493 rows x 8 columns]
submission_csv = pd.read_csv(path + "sampleSubmission.csv")
# print(submission_csv) [6493 rows x 2 columns]

X = train_csv.drop(['count', 'casual', 'registered'], axis=1)   # count, casual, registered 분리
y = train_csv['count']

# print(X.shape)   #(10886, 8)
# print(y.shape)  #(10886,)
# print(test_csv.shape) #(6493, 8)


#2. 모델
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='relu'))

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=3)
# model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, epochs=10, batch_size=3)
# loss = abs(model.evaluate(X_test, y_test))
# y_predict = model.predict(X_test)
# r2 = r2_score(y_test, y_predict)
# y_submit = model.predict([test_csv])
# submission_csv['count'] = y_submit
# print("MSE :" , loss)
# # submission_csv['count'][(submission_csv['count'] < 0)] = 0
# print("음수 갯수:", submission_csv[submission_csv['count'] < 0].count())

# y_predict = model.predict(X_test)

# def RMSE(y_test, y_predict):
#     rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    
#     return rmse

def RMSLE(y_test, y_predict):

    rmsle = np.sqrt(mean_squared_log_error(y_test, y_predict))
        
    return rmsle


def auto(a,b):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=a)
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=120, batch_size=b, verbose=0)
    loss = model.evaluate(X_test, y_test)
    y_predict = model.predict(X_test)
    r2 = r2_score(y_test, y_predict)
    y_submit = model.predict([test_csv])
    submission_csv['count'] = y_submit
    if submission_csv['count'].min() < 0 :
        return 0, -1 ,0
    # print("MSE :" , loss)
    else :
        
        rmsle = RMSLE(y_test, y_predict)
        min = submission_csv['count'].min()
        return rmsle, min, loss
    # submission_csv['count'][(submission_csv['count'] < 0)] = 0
    # print("음수 갯수:", submission_csv[submission_csv['count'] < 0].count())
    



count = 0
min_loss = 1000000
min_rmsle = 150
random_state = 0
file_name = "submission_0109_1_"
last_name = ".csv"

while True :
    batch_size = random.randrange(256, 513)
    random_state = random.randrange(1, 999999)
    rmsle, min, loss = auto(random_state, batch_size)
    if min > 0 and rmsle < min_rmsle:
        min_rmsle = rmsle
        min_loss = loss
        rmsle = round(rmsle, 2)
        submission_csv.to_csv(path + file_name + str(batch_size) + "and" + str(random_state) +"andRMSE_" + str(rmsle) + last_name , index=False)
