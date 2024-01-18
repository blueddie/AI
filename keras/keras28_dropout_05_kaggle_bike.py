# https://www.kaggle.com/competitions/bike-sharing-demand/leaderboard
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import random
import time, datetime

#1. 데이터
csv_path = "C://_data//kaggle//bike//"
train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sampleSubmission.csv")

X = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']
test_csv = test_csv.drop([], axis=1)

#2. 모델
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

def RMSLE(y_test, y_predict):
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_predict))
    
    return rmsle

def ACC(aaa, bbb):
    return accuracy_score(aaa,bbb)

date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   
path = '..\\_data\_save\\MCP\\kaggle_bike\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'kaggle_bike_', date, '_' ,filename])

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=50
                   , verbose=1
                   , restore_best_weights=True
                   )

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=15)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)
test_csv = scaler.transform(test_csv)


model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_train, y_train, epochs=1200, batch_size=32
                    , validation_split=0.2 
                    , callbacks=[es, mcp]
                    )

y_predict = model.predict(X_test)

loss = model.evaluate(X_test, y_test)
rmsle = RMSLE(y_test, y_predict)
# r2 = r2_score(y_test, y_predict)
y_submit = model.predict([test_csv])
submission_csv['count'] = y_submit

print('rmsle : ' , rmsle)
print('loss : ' , loss)

submission_csv.to_csv(csv_path + date + str(round(rmsle, 5)) +".csv", index=False)