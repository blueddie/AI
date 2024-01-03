import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([range(10)])
x = x.T
print(x)    # [[0 1 2 3 4 5 6 7 8 9]]  (1, 10)
print(x.shape) #(10, 1)

y = np.array([[1,2,3,4,5,6,7,8,9,10]
             , [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
             , [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
             ]) 
y = y.T
print(y)
print(y.shape)
# print(y)
# #2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))

# # #3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=5000, batch_size=3)
# # #4. 평가, 예측
loss = model.evaluate(x, y)
results = model.predict([[10]])
print("loss : " , loss)
print("예측 값은 : " , results)

# Epoch 5000/5000
# 4/4 [==============================] - 0s 0s/step - loss: 3.4942e-10
# 1/1 [==============================] - 0s 58ms/step - loss: 3.3778e-10
# 1/1 [==============================] - 0s 50ms/step
# loss :  3.377811086391347e-10
# 예측 값은 :  [[11.         1.9999999 -0.9999676]]


