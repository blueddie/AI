# https://www.kaggle.com/competitions/bike-sharing-demand/leaderboard
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import random
import time, datetime
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
#1. 데이터
csv_path = "C://_data//kaggle//bike//"
train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sampleSubmission.csv")

x = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']

# print(x.shape)  #(10886, 10)
# print(pd.isna(x).sum()) # 결측치 없음
# print(pd.isna(test_csv).sum())  # 결측치 없음

# print(x.dtypes)
# print(test_csv.dtypes)
x = x.astype('float32')
test_csv = test_csv.astype('float32')

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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=15)
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# print(test_csv.shape)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델
model = Sequential()
model.add(Dense(8, input_shape=(8,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

def RMSLE(y_test, y_predict):
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_predict))
    return rmsle
 
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=100
                   , verbose=1
                   , restore_best_weights=True
                   )

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

hist = model.fit(x_train, y_train, epochs=4000, batch_size=55, validation_split=0.2, callbacks=[es])

y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test)
rmsle = RMSLE(y_test, y_predict)

y_submit = model.predict(test_csv)

submission_csv['count'] = y_submit

print(f"rmsle : {rmsle}")
print(f"loss : {loss}")

date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053  
submission_csv.to_csv(csv_path + date + ".csv", index=False)

# rmsle : 1.3056097520463865
# loss : 21516.712890625