import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 고의적으로 R2 값 낮추기.
# 1. R2를 음수가 아닌 0.5 미만으로 만들 것
# 2. 데이터는 건들지 말 것.
# 3. 레이어는 인풋과 아웃풋 포함해서 7개 이상
# 4. batch_size=1
# 5. 히든 레이어의 노드는 10개 이상 100개 이하
# 6.train 사이즈 75
# epochs 100번 이상#
#
# 

#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=333)

#2
model = Sequential()
model.add(Dense(97,input_dim=1))
model.add(Dense(98))
model.add(Dense(96))
model.add(Dense(95))
model.add(Dense(35))
model.add(Dense(10))
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

# Epoch 100/100
# 15/15 [==============================] - 0s 0s/step - loss: 12.6615        
# 1/1 [==============================] - 0s 67ms/step - loss: 7.7114
# 1/1 [==============================] - 0s 71ms/step
# 1/1 [==============================] - 0s 17ms/step
# R2 score :  0.016398054769862225

# Epoch 100/100
# 15/15 [==============================] - 0s 0s/step - loss: 13.9306        
# 1/1 [==============================] - 0s 66ms/step - loss: 7.7606
# 1/1 [==============================] - 0s 72ms/step
# 1/1 [==============================] - 0s 17ms/step
# R2 score :  0.010128965761240671
# loss :  7.760588645935059

# Epoch 100/100
# 15/15 [==============================] - 0s 434us/step - loss: 16.7801     
# 1/1 [==============================] - 0s 71ms/step - loss: 7.7913
# 1/1 [==============================] - 0s 55ms/step
# 1/1 [==============================] - 0s 3ms/step
# R2 score :  0.006210001156924494
# loss :  7.791313171386719