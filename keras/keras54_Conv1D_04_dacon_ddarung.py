# https://dacon.io/competitions/open/235576/codeshare/6969?page=1&dtype=recent // 대회 주소
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error
from sklearn.linear_model import LinearRegression
import random, _bootlocale
import time
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler


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


X = X.astype(np.float32)
test_csv = test_csv.astype(np.float32)




#2. 모델
model = Sequential()
model.add(Conv1D(8,2, input_shape=(9,1), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=150
                   , verbose=1
                   , restore_best_weights=True
                   )
def RMSE(y_test, y_predict):
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    
    return rmse


    # rs = 56238592
    # bs = 47
    # epo = random.randrange(100, 9999)
    
    
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.86, random_state=56238592)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
test_csv = scaler.transform(test_csv)

X_train = X_train.reshape(-1, 9, 1)
X_test = X_test.reshape(-1, 9, 1)
test_csv = test_csv.reshape(-1, 9, 1)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#3
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=1000, batch_size=47
            , validation_split=0.2
            , callbacks=[es]
            )
#4. 평가,예측
loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
rmse = RMSE(y_test, y_predict)

y_submit = model.predict([test_csv])

submission_csv['count'] = y_submit

import datetime
date = datetime.datetime.now().strftime("%m%d_%H%M")    
    
submission_csv.to_csv(path + date + '_'+ str(round(rmse, 3)) + ".csv", index=False)
   

print("loss : ", loss)
print("rmse : " , rmse)

# [6708.2822265625, 0.0034246575087308884]  scaleX
# [6410.08203125, 0.0]    scale

# loss :  1436.9796142578125
# rmse :  0.40732259246917946