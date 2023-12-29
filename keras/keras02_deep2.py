from keras.layers import Dense
from keras.models import Sequential
import numpy as np

#1 데이터

x = np.array([1,2,3,4,5,6])

y = np.array([1,2,3,5,4,6])

# Epoch 100/100
# 1/1 [==============================] - 0s 0s/step - loss: 0.3257
# 1/1 [==============================] - 0s 78ms/step - loss: 0.3255
# loss :  0.325509250164032
# 1/1 [==============================] - 0s 52ms/step
# 7의 예측 값은 :  [[6.710495]]

# 위 데이터를 훈련해서 최소의 loss

#2 모델 구성
#### [실습] 100 epoch 01_1번과 같은 결과를 얻어라

model = Sequential()
model.add(Dense(16, input_dim=1))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(1))



#3 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

#4 평가, 예측

loss = model.evaluate(x, y)
print("loss : " , loss)
result = model.predict([7])
print("7의 예측 값은 : " , result)

# loss :  0.3238094747066498
# 1/1 [==============================] - 0s 34ms/step
# 7의 예측 값은 :  [[1.1428571]
#  [2.0857143]
#  [3.0285714]
#  [3.9714286]
#  [4.9142857]
#  [5.857143 ]
#  [6.8      ]]