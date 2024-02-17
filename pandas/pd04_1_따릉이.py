# https://dacon.io/competitions/open/235576/codeshare/6969?page=1&dtype=recent // 대회 주소
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import random
import time
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "submission.csv")   #, index_col=0

xy = train_csv.copy()

x = xy.drop(['count'], axis=1)
y = xy['count']

# print(x.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5'],
#       dtype='object')

# print(pd.isna(x).sum())
# hour                        0
# hour_bef_temperature        2
# hour_bef_precipitation      2
# hour_bef_windspeed          9
# hour_bef_humidity           2
# hour_bef_visibility         2
# hour_bef_ozone             76
# hour_bef_pm10              90
# hour_bef_pm2.5            117

# print(pd.isna(test_csv).sum())
# hour                       0
# hour_bef_temperature       1
# hour_bef_precipitation     1
# hour_bef_windspeed         1
# hour_bef_humidity          1
# hour_bef_visibility        1
# hour_bef_ozone            35
# hour_bef_pm10             37
# hour_bef_pm2.5            36

x = x.astype('float32')
test_csv = test_csv.astype('float32')
# 결측치 처리
x = x.interpolate() 
test_csv = test_csv.interpolate()

# 결측치 없음을 확인
# print(pd.isna(test_csv).sum())
# print(pd.isna(x).sum())
# hour                      0
# hour_bef_temperature      0
# hour_bef_precipitation    0
# hour_bef_windspeed        0
# hour_bef_humidity         0
# hour_bef_visibility       0
# hour_bef_ozone            0
# hour_bef_pm10             0
# hour_bef_pm2.5            0

def outlierHandler(data):
    data = pd.DataFrame(data)
    
    for label in data:
        series = data[label]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        
        iqr = q3 - q1
        upper_bound = q3 + iqr
        lower_bound = q1 - iqr
        
        series[series > upper_bound] = np.nan
        series[series < lower_bound] = np.nan
        
        # print(series.isna().sum())
        series = series.interpolate()
        data[label] = series
        
        data = data.fillna(data.ffill())
        data = data.fillna(data.bfill())

    return data

x = outlierHandler(x)
test_csv = outlierHandler(test_csv)
# 이상치 nan 처리 후 interpolate
# print(pd.isna(x).sum())
# print(pd.isna(test_csv).sum())

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1226)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

#2 모델
model = Sequential()
model.add(Dense(8, input_shape=(9,), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))


#3 컴파일 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=110
                   , verbose=1
                   , restore_best_weights=True
                   )

def RMSE(y_test, y_predict):
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    return rmse

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
hist = model.fit(x_train, y_train, epochs=4000, batch_size=16, validation_split=0.2, verbose=2, callbacks=[es])


#4 평가, 예측
loss = model.evaluate(x_test,y_test,verbose=0)
y_predict = model.predict(x_test,verbose=0)

rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

y_submit = model.predict(test_csv,verbose=0)
submission_csv['count'] = y_submit

import datetime
date = datetime.datetime.now().strftime("%m%d_%H%M")
submission_csv.to_csv(path + date + ".csv", index=False)

print(f"rmse : {rmse}")
print(f"r2 : {r2}")

# x 만 이상치 처리
# rmse : 44.63514138226922
# r2 : 0.7274054751313268

# x, test_csv 모두 이상치 처리
# rmse : 48.1338346163343
# r2 : 0.6829963796991915

