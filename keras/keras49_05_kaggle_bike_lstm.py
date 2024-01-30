# https://www.kaggle.com/competitions/bike-sharing-demand/leaderboard
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import random
import time

#1. 데이터
path = "C://_data//kaggle//bike//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")

X = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']
test_csv = test_csv.drop([], axis=1)

print(X.shape)  #(10886, 8)
print(test_csv.shape)



#2. 모델
model = Sequential()
model.add(LSTM(120, input_shape=(8,1), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dense(55, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

def RMSLE(y_test, y_predict):
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_predict))
    
    return rmsle

def ACC(aaa, bbb):
    return accuracy_score(aaa,bbb)


from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=50
                   , verbose=1
                   , restore_best_weights=True
                   )



x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1334)





from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


x_train = x_train.reshape(-1,8 ,1)
x_test = x_test.reshape(-1, 8, 1)
test_csv = test_csv.reshape(-1,8,1 )


model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=1200, batch_size=32
                    , validation_split=0.2 
                    , callbacks=[es]
                    )

y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test)
rmsle = RMSLE(y_test, y_predict)
# r2 = r2_score(y_test, y_predict)
y_submit = model.predict([test_csv])
submission_csv['count'] = y_submit
    # print("rmsle", rmsle)
    # print("loss", loss[0])
    

import datetime
date = datetime.datetime.now().strftime("%m%d_%H%M")    

extension_name = ".csv"

# rmsle, rs, bs, loss, hist = auto()

min_rmsle = rmsle
submission_csv.to_csv(path + "and"+ str(rmsle) + extension_name, index=False)
