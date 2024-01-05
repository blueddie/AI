# https://dacon.io/competitions/open/235576/codeshare/6969?page=1&dtype=recent // 대회 주소

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path + "submission.csv")   #, index_col=0
print(submission_csv)

print(train_csv.shape)  #(1459, 10)
print(test_csv.shape)   #(715, 9)
print(submission_csv.shape) #(715, 2)

# print(train_csv.columns)
# # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
# #        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
# #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'
# print(train_csv.info())
# print(test_csv.info())

print(train_csv.info()) #std 표준편차, 50% 중위값
print(test_csv.info())


#################결측치 제거###################
# print(train_csv.isnull().sum())
print(train_csv.isna().sum())   #위 코드와 같다.
train_csv = train_csv.dropna()  # 한 행에 결측기가 하나라도 있다면 그 행 전체를 삭제!!!
print(train_csv.isna().sum())   #위 코드와 같다.
print(train_csv.info())
print(train_csv.shape)  #(1328, 10)

test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())


#2. 모델
model = Sequential()
model.add(Dense(8, input_dim=9))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

########## x , y 분리 #############

x = train_csv.drop(["count"], axis=1)
print(x)
y = train_csv["count"]
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=3)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#3
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=5)

#4. 평가,예측
loss = model.evaluate(x_test, y_test)

y_submit = model.predict([test_csv])

print(loss)
# print(y_submit)
# print(y_submit.shape)


############### submission.csv 만들기 (count column에 값만 넣어주면 됨) ################
submission_csv['count'] = y_submit
print(submission_csv)
# print(submission_csv.shape)

submission_csv.to_csv(path + "submission_0003.csv", index=False)
