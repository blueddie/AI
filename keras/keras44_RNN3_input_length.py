import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3]
             , [2,3,4]
             , [3,4,5]
             , [4,5,6]
             , [5,6,7]
             , [6,7,8]
             , [7,8,9]])

y= np.array([4, 5 ,6, 7, 8, 9, 10])

print(x.shape)  #(7, 3)
print(y.shape)  #(7,)

x = x.reshape(7, 3, 1)
print(x.shape)  #(7, 3, 1)

#2 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=16,activation='relu', input_shape=(3,1))) #timesteps, features
model.add(SimpleRNN(10, input_length=3, input_dim=1))
# model.add(SimpleRNN(10, input_dim=1 ,input_length=3 ))    #가독성이 떨어짐
# 3-D tensor with shape (batch_size, timesteps, features)
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

#3 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=7000)

#4
results = model.evaluate(x, y)
print('loss : ', results)
y_predict = np.array([8,9,10])
y_predict = y_predict.reshape(1,3,1)
y_predict  = model.predict(y_predict)
# (3,) -> (1, 3, 1) 
print('예측 값은 : ',  y_predict)

#예측 값은 :  [[10.815441]]
# 예측 값은 :  [[10.922621]]
# 예측 값은 :  [[11.000694]]