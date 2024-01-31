import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM , Conv1D, Flatten

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3]
             , [2,3,4]
             , [3,4,5]
             , [4,5,6]
             , [5,6,7]
             , [6,7,8]
             , [7,8,9]])

y = np.array([4, 5 ,6, 7, 8, 9, 10])

print(x.shape)  #(7, 3)
print(y.shape)  #(7,)

x = x.reshape(7, 3, 1)
print(x.shape)  #(7, 3, 1)

#2 모델 구성
model = Sequential()
# model.add(LSTM(units=10,activation='relu', input_shape=(3,1))) #timesteps, features
# 3-D tensor with shape (batch_size, timesteps, features)
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(3,1)))
model.add(Flatten())
model.add(Dense(7))
model.add(Dense(1))

model.summary()

# LSTM : Total params: 565
# conv1d :Total params: 185


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

# 예측 값은 :  [[11.000694]]

# Conv1D
# loss :  2.5985564069443134e-13
# 예측 값은 :  [[10.999999]]