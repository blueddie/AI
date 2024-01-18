# https://dacon.io/competitions/open/235576/codeshare/6969?page=1&dtype=recent // 대회 주소
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
csv_path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
print(train_csv)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
print(test_csv)
submission_csv = pd.read_csv(csv_path + "submission.csv")   #, index_col=0
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


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1226)
#2. 모델
model = Sequential()
model.add(Dense(8, input_dim=9, activation='relu'))
model.add(Dense(16))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

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

import datetime

date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   
path = '..\\_data\_save\\MCP\\dacon_bike\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'dacon_bike_', date, '_' ,filename])




mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

scaler = StandardScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)
test_csv = scaler.transform(test_csv)

#3

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1000, batch_size=16
            , validation_split=0.2
            , callbacks=[es, mcp]
            )


#4. 평가,예측
loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
rmse = RMSE(y_test, y_predict)

y_submit = model.predict([test_csv])

submission_csv['count'] = y_submit

print("loss : ", loss)
print("rmse : " , rmse)

submission_csv.to_csv(csv_path + date + str(round(rmse, 5)) +".csv", index=False)

# loss :  6923.4462890625
# rmse :  83.20724884884876