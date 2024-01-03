import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

#2. 
model = Sequential()
model.add(Dense(1, input_dim=1))

#3.
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=3000, batch_size=3)

#4.
loss = model.evaluate(x, y)
results = model.predict([11000, 7])
print("loss : ", loss)
print("예측값은 : ", results)