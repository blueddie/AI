import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Bidirectional, GRU

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
model.add(GRU(units=10, input_shape=(3,1)))
model.add(Dense(7,activation='relu'))
model.add(Dense(1))

model.summary()

'''
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
'''

# loss :  1.8355128972302737e-08
# 1/1 [==============================] - 0s 117ms/step
# 예측 값은 :  [[10.524493]]

