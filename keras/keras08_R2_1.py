import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=412)

#2
model = Sequential()
model.add(Dense(8,input_dim=1))
model.add(Dense(16))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

# #4
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
results = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 score : ", r2)

print("loss : ", loss )


plt.scatter(x,y)
plt.plot(x, results, color='red')
plt.show()

# random_state= 412

# Epoch 100/100
# 14/14 [==============================] - 0s 0s/step - loss: 16.8771        
# 1/1 [==============================] - 0s 94ms/step - loss: 1.0862
# 1/1 [==============================] - 0s 35ms/step
# 1/1 [==============================] - 0s 0s/step
# R2 score :  0.9636599622515736
# loss :  1.0861634016036987

# Epoch 100/100
# 14/14 [==============================] - 0s 0s/step - loss: 16.8590        
# 1/1 [==============================] - 0s 64ms/step - loss: 1.0480
# 1/1 [==============================] - 0s 35ms/step
# 1/1 [==============================] - 0s 2ms/step
# R2 score :  0.9649363245314011
# loss :  1.0480142831802368
# ---------------------------------------------------------
# model.add(Dense(50))
# model.add(Dense(20))
# model.add(Dense(15))
# Epoch 100/100
# 14/14 [==============================] - 0s 1ms/step - loss: 17.1287
# 1/1 [==============================] - 0s 81ms/step - loss: 0.9907
# 1/1 [==============================] - 0s 81ms/step
# 1/1 [==============================] - 0s 15ms/step
# R2 score :  0.9668538291654396
# loss :  0.9907021522521973