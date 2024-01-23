# https://dacon.io/competitions/open/235576/codeshare/6969?page=1&dtype=recent // 대회 주소
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
csv_path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
# print(train_csv)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
# print(test_csv)
submission_csv = pd.read_csv(csv_path + "submission.csv")   #, index_col=0
# print(submission_csv)

X = train_csv.drop(['count'], axis=1)
y = train_csv["count"]
# print(X.shape)  #(1459, 9)
# print(y.shape)  #(1459,)

test_csv = test_csv.fillna(test_csv.mean())
X = X.fillna(X.mean())
X['hour_bef_temperature'] = X['hour_bef_temperature'].fillna(X['hour_bef_temperature'].mean())

X.loc[[1420],['hour_bef_precipitation']] = 0.0
X.loc[[1553],['hour_bef_precipitation']] = 1.0

X.loc[[1420],['hour_bef_humidity']] = 37.0
X.loc[[1553],['hour_bef_humidity']] = 82.0
X['hour_bef_visibility'] = X['hour_bef_visibility'].fillna(X['hour_bef_visibility'].mean())


X = X.values.reshape(-1, 3, 3, 1)   #(1459, 3, 3, 1)
# print(X.shape)  #(1459, 9)
# print(test_csv.shape)   #(715, 9)
test_csv = test_csv.values.reshape(-1, 3, 3, 1)
# print(test_csv.shape)



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=1226)

X_train = np.asarray(X_train).astype(np.float32) 
X_test = np.asarray(X_test).astype(np.float32)
test_csv = np.asarray(test_csv).astype(np.float32)

X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)
test_csv_flattened = test_csv.reshape(test_csv.shape[0], -1)


scaler = StandardScaler()
scaler.fit(X_train_flattened)
scaled_train = scaler.transform(X_train_flattened)
scaled_test = scaler.transform(X_test_flattened)
scaled_test_csv = scaler.transform(test_csv_flattened)

X_train = scaled_train.reshape(X_train.shape)
X_test = scaled_test.reshape(X_test.shape)
test_csv = scaled_test_csv.reshape(test_csv.shape)



# X_test = scaled_test.reshape(X_test.shape)

#2
model = Sequential()
model.add(Conv2D(97, (2,2), activation='relu', padding='same', input_shape=(3, 3, 1)))
model.add(MaxPooling2D(strides=(1, 1)))
model.add(Conv2D(32, (1, 2), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(Conv2D(160, (1,2), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(Conv2D(120, (1,2), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3)) # 방금 추가
model.add(Dense(23, activation='relu'))
model.add(Dense(1)) 

def RMSE(y_test, y_predict):
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    
    return rmse

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=30
                   , verbose=1
                   , restore_best_weights=True
                   )

import datetime
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1000, batch_size=16
            , validation_split=0.2
            , callbacks=[es]
            )

#4
loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
rmse = RMSE(y_test, y_predict)

y_submit = model.predict([test_csv])
submission_csv['count'] = y_submit

print("loss : ", loss)
print("rmse : " , rmse)

submission_csv.to_csv(csv_path + str(round(rmse, 5)) +".csv", index=False)

