# https://dacon.io/competitions/open/235576/codeshare/6969?page=1&dtype=recent // 대회 주소

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import random
import time

#1. 데이터    
path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path + "submission.csv")   #, index_col=0
print(submission_csv)

X = train_csv.drop(['count', 'hour_bef_ozone', 'hour_bef_pm2.5','hour_bef_pm10'], axis=1)

y = train_csv["count"]

test_csv = test_csv.drop(['hour_bef_ozone', 'hour_bef_pm2.5','hour_bef_pm10'], axis=1)
test_csv = test_csv.fillna(test_csv.mean())


X['hour_bef_temperature'] = X['hour_bef_temperature'].fillna(X['hour_bef_temperature'].mean())

X.loc[[1420],['hour_bef_precipitation']] = 0.0
X.loc[[1553],['hour_bef_precipitation']] = 1.0

X.loc[[1420],['hour_bef_humidity']] = 37.0
X.loc[[1553],['hour_bef_humidity']] = 82.0

X['hour_bef_visibility'] = X['hour_bef_visibility'].fillna(X['hour_bef_visibility'].mean())
X['hour_bef_windspeed'] = X['hour_bef_windspeed'].fillna(X['hour_bef_windspeed'].mean())

print(test_csv.info())

first_name = "submission_0109_2"
second_name = ".csv"
i = 1
haha = 0

#2. 모델
model = Sequential()
model.add(Dense(8, input_dim=6, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))
    
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=53145880)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#3
model.compile(loss='mse', optimizer='adam')
start_time =time.time()

hist = model.fit(X_train, y_train, epochs=10, batch_size=32
            , validation_split=0.2)
end_time = time.time()   
    
    
         
#4. 평가,예측
loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)

y_submit = model.predict([test_csv])
print("loss : ",loss)
print("r2 : ", r2)

submission_csv['count'] = y_submit
print("R2 score : ", r2)
print("loss : " , loss)
print("소요 시간 : ", round(end_time - start_time, 2), "seconds")

print("======================== hist ===========================================")
print(hist) # wrapping 된 상태
print("========================== hist.history =========================================")
print(hist.history)
print("========================= loss ==============================")
print(hist.history['loss'])
print("========================= val_loss ==============================")
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   #한글 깨짐 해결
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss', marker='.')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')
plt.legend(loc='upper right')
plt.title('따릉이 loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()  
plt.show()
