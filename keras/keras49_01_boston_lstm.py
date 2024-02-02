from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, SimpleRNN, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
import datetime

datasets = load_boston()

x = datasets.data
y = datasets.target
# print(x.shape)  #(506, 13)
x = x.reshape(-1 , 13, 1)
# print(x.shape)  #(506, 13, 1)
print(y.shape)  #(506,)



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.8)
print(x_train.shape)    
print(x_test.shape)     
print(y.shape)         

#2
model = Sequential()
model.add(LSTM(16, activation='relu',input_shape=(13,1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))


#3
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=40
                   , verbose=1
                   , restore_best_weights=True
                   )

model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=64, validation_split=0.2, callbacks=[es])

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
results = model.predict(x)

r2 = r2_score(y_test, y_predict)

print("R2 score : ", r2)
print("loss : " , loss)

# R2 score :  0.7615445774567593
# loss :  3.145082473754883

# R2 score :  0.6817796619773973
# loss :  3.125746965408325

# R2 score :  0.6736491113441989
# loss :  3.5278685092926025

# R2 score :  0.7359853474011155
# loss :  2.8940417766571045


#LSTM
# R2 score :  0.6397616496413631
# loss :  3.9518702030181885

# R2 score :  0.7228187387717201
# loss :  3.4327473640441895


