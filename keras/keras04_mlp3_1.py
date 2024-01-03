import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([range(10)])
print(x)    #[[0 1 2 3 4 5 6 7 8 9]]
print(x.shape)  #(1, 10)

x = np.array([range(1, 10)])
print(x)    #[[1 2 3 4 5 6 7 8 9]]
print(x.shape)  #(1, 9)

#  위 내용은 range의 이해를 돕기 위함

x = np.array([range(10), range(21, 31), range(201, 211)])
print(x)
print(x.shape) 
x = x.T
print(x)

print(x.shape)


y = np.array([[1,2,3,4,5,6,7,8,9,10]
             , [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
             ])    # python []안에 있는 2개 이상은 ★ list ★
y = y.T
print(y.shape)

# 2. 모델 구성
model = Sequential()
model.add(Dense(2, input_dim=3))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=3, epochs=4000)

#4. 평가, 예측
#   실습[10, 31, 211]
loss = model.evaluate(x, y)
results = model.predict([[10,31,211]])
print("loss : " , loss)
print("예측 값은 : ", results)

# Epoch 800/800
# 4/4 [==============================] - 0s 515us/step - loss: 5.1072e-11
# 1/1 [==============================] - 0s 80ms/step - loss: 3.5447e-11
# 1/1 [==============================] - 0s 69ms/step
# loss :  3.54468468466429e-11
# 예측 값은 :  [[11.000013   2.0000083]]

# Epoch 4000/4000
# 4/4 [==============================] - 0s 0s/step - loss: 6.8070e-13
# 1/1 [==============================] - 0s 62ms/step - loss: 6.7786e-13
# 1/1 [==============================] - 0s 36ms/step
# loss :  6.778577807571573e-13
# 예측 값은 :  [[10.999999   1.9999986]]

# Epoch 4000/4000
# 4/4 [==============================] - 0s 0s/step - loss: 1.4122e-11
# 1/1 [==============================] - 0s 49ms/step - loss: 1.1568e-12
# 1/1 [==============================] - 0s 57ms/step
# loss :  1.1567635955014866e-12
# 예측 값은 :  [[11.         2.0000007]]
