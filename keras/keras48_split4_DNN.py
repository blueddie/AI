import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN, LSTM
from keras.callbacks import EarlyStopping

#1
a = np.array(range(1, 101))

x_predict = np.array(range(96,106))

size = 5    # x 데이터는 4개 y 데이터는 1개
# print(len(a))

# print(x_predict)
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1) :
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

xy = split_x(a, size)
# print(xy)
x = xy[:, :-1]
y = xy[:, -1]
x_predict = split_x(x_predict, (size - 1))

# print(x.shape)  #(96, 4)
# print(y.shape)  #(96,)

#2
model = Sequential()
model.add(Dense(units=8,  activation='relu', input_shape=(4,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3
es = EarlyStopping(monitor='loss', mode='min', patience=40, verbose=1, restore_best_weights=True)
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, callbacks=[es], batch_size=8)

#4
results = model.evaluate(x, y)
print('loss : ', results)
y_predict  = model.predict(x_predict)
print('예측 값은 : ',  y_predict)

