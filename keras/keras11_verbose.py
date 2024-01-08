import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,6,5,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,6,5,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 
model = Sequential()
model.add(Dense(1, input_dim=1))

#3.
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=3, 
          verbose=0)
# verbose=0 : 침묵
# verbose=1 : default
# verbose=2 : 프로그래스바 삭제
# verbose=3 : 에포만 출력
# verbose=4 : 

#4.
loss = model.evaluate(x_test, y_test)
results = model.predict([11000, 7])
print("loss : ", loss)
print("예측값은 : ", results)