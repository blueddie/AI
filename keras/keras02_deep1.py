from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1 data
x = np.array([1,2,3])

y = np.array([1,2,3])

#2 모델 구성
model = Sequential()
model.add(Dense(16, input_dim=1))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(1))

# Epoch 100/100
# 1/1 [==============================] - 0s 0s/step - loss: 6.8195e-05
# 1/1 [==============================] - 0s 77ms/step - loss: 5.8538e-05
# loss :  5.853833499713801e-05
# 1/1 [==============================] - 0s 60ms/step
# 4의 예측 값은 :  [[4.000446]]

# Epoch 100/100
# 1/1 [==============================] - 0s 0s/step - loss: 1.3399e-04
# 1/1 [==============================] - 0s 76ms/step - loss: 1.3241e-04
# loss :  0.00013241189299151301
# 1/1 [==============================] - 0s 71ms/step
# 4의 예측 값은 :  [[3.9999838]]


#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

#4  평가, 예측
loss = model.evaluate(x, y)
print('loss : ' , loss)
result = model.predict([4])
print("4의 예측 값은 : " , result)