#06_1 카피
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,6,5,7,8,9,10])

x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,6])

x_val = np.array([6,7])
y_val = np.array([5,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3.
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=3100, batch_size=3
          ,validation_data=(x_val, y_val))

#4.
loss = model.evaluate(x_test, y_test)
results = model.predict([11000, 7])
print("loss : ", loss)
print("예측값은 : ", results)