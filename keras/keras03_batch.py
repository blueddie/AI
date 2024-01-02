# from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import keras 

print('tf version :',tf.__version__)
print("keras version : ", keras.__version__)

#1. 데이터
x = np.array([1,2,3,4,5,6])

y = np.array([1,2,3,5,4,6])

#2. 모델 구성
model = Sequential()
model.add(Dense(16, input_dim=1))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3.  컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=4) # batch_size 기본 값은 32

#4. 평가, 예측
loss = model.evaluate(x, y)
results = model.predict([7])
print("loss : ", loss)
print("7의 예측 값은 : ", results)




# model = Sequential()
# model.add(Dense(16, input_dim=1))
# model.add(Dense(32))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(2))
# model.add(Dense(1))

# batch_size=3
# Epoch 100/100
# 1/2 [==============>...............] - ETA: 0s - loss: 0.6412/2 [==============================] - 0s 0s/step - loss: 0.3271
# 1/1 [==============================] - ETA: 0s - loss: 0.3241/1 [==============================] - 0s 91ms/step - loss: 0.3248
# 1/1 [==============================] - 0s 85ms/step
# loss :  0.3247807025909424
# 7의 예측 값은 :  [[6.861649]]

# batch_size=2
# Epoch 100/100
# 1/3 [=========>....................] - ETA: 0s - loss: 0.4383/3 [==============================] - 0s 0s/step - loss: 0.3495
# 1/1 [==============================] - ETA: 0s - loss: 0.3231/1 [==============================] - 0s 84ms/step - loss: 0.3238
# 1/1 [==============================] - 0s 69ms/step
# loss :  0.3238222301006317
# 7의 예측 값은 :  [[6.799369]]
#
