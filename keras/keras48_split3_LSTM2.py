import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN, LSTM
from keras.callbacks import EarlyStopping


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
# print('a : ' , a)
# print('a shape : ', a.shape)

# bbb = split_x(a, size)
# print('bbb: ' ,bbb)
# print('bbb shape : ',bbb.shape)

x = xy[:, :-1]
y = xy[:, -1]

# print(x)
# print(y)
# print(x.shape)  #(96, 4)
# print(y.shape)  #(96,)
x_predict = split_x(x_predict, (size - 1))
# print(x_predict.shape)  #(7, 4)
# print(x_predict)

x = x.reshape(-1, 1, 4)
x_predict = x_predict.reshape(-1, 1, 4)
# print(x.shape)  #(96, 4, 1)
# print(x_predict.shape)  #(7, 4, 1)

#2
model = Sequential()
model.add(LSTM(units=8,  activation='relu', input_shape=(1,4))) #timesteps, features
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3
es = EarlyStopping(monitor='loss', mode='min', patience=4000, verbose=1, restore_best_weights=True)
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=20000, callbacks=[es], batch_size=8)

#4
results = model.evaluate(x, y)
print('loss : ', results)
y_predict  = model.predict(x_predict)
print('예측 값은 : ',  y_predict)

# loss :  0.03194255381822586
# 예측 값은 :  [[ 99.79366 ]
#  [100.668045]
#  [101.52809 ]
#  [102.3733  ]
#  [103.20312 ]
#  [104.01711 ]
#  [104.81485 ]]



# loss :  0.055988650768995285
# 예측 값은 :  [[ 99.23254 ]
#  [100.05349 ]
#  [100.859825]
#  [101.65126 ]
#  [102.42751 ]
#  [103.188385]
#  [103.93367 ]]

# 예측 값은 :  [[100.02989 ]
#  [101.03386 ]
#  [102.037994]
#  [103.042274]
#  [104.0467  ]
#  [105.051254]
#  [106.05593 ]]





# (N, 1, 4)
# loss :  2.0551263815726806e-11
# 1/1 [==============================] - 0s 102ms/step
# 예측 값은 :  [[100.      ]
#  [101.000015]
#  [102.      ]
#  [103.      ]
#  [104.      ]
#  [105.00001 ]
#  [106.000015]]