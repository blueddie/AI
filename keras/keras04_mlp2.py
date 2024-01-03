#[실습]
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10]
             , [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]
             , [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
             ]
             )

y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x)
print(x.shape, y.shape) #(2,10) (10,) (열, 컬럼, 차원)
x = x.T
print(x.shape, y.shape) #(10,2)

#2. 모델 구성
model = Sequential()
model.add(Dense(8, input_dim=3))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1550, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x, y)
results = model.predict([[10,1.3,0]])

print("loss : " , loss)
print("예측 값은 : " , results)

# Epoch 1200/1200
# 1/4 [======>.......................] - ETA: 0s - loss: 6.134/4 [==============================] - 0s 6ms/step - loss: 3.4973e-12
# 1/1 [==============================] - ETA: 0s - loss: 6.341/1 [==============================] - 0s 66ms/step - loss: 6.3409e-12
# 1/1 [==============================] - 0s 57ms/step
# loss :  6.340883547395482e-12
# 예측 값은 :  [[9.999995]]

# Epoch 1650/1650
# 1/4 [======>.......................] - ETA: 0s - loss: 2.8043e-4/4 [==============================] - 0s 0s/step - loss: 1.8659e-12
# 1/1 [==============================] - ETA: 0s - loss: 1.2008e-1/1 [==============================] - 0s 62ms/step - loss: 1.2008e-12
# 1/1 [==============================] - 0s 51ms/step
# loss :  1.2008172234345693e-12
# 예측 값은 :  [[10.000003]]

# Epoch 1550/1550
# 4/4 [==============================] - 0s 0s/step - loss: 9.5156e-12
# 1/1 [==============================] - 0s 83ms/step - loss: 6.9647e-12
# 1/1 [==============================] - 0s 50ms/step
# loss :  6.964739982656676e-12
# 예측 값은 :  [[10.000002]]