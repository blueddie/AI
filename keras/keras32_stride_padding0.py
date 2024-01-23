import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(X_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

# print(X_train)
# print(X_train[0])
# print(y_train[0])   #5
# print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))
# print(pd.value_counts(y_train))

X_train = X_train.reshape(60000, 28, 28, 1)
# X_test = X_test.reshape(10000, 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

print(X_test.shape)

#2
model = Sequential()
model.add(Conv2D(9, (2,2) , strides=1
                 , padding='same' # default : valid (2,2)
                 , input_shape=(5,5,1))
                 #input_shape = (batch_size, row, columns, chanels)
                 #input_shape = (batch_size, height, width, chanels)
          )
model.add(Conv2D(filters=10, kernel_size=(3,3))) 
model.add(Conv2D(15, (4,4)))
model.add(Flatten())
model.add(Dense(units=8))
model.add(Dense(7, input_shape=(8,)))
#                   shape=(batch_size, input_dim)
model.add(Dense(6))
model.add(Dense(10, activation='softmax'))

model.summary()

# (kernal_size * kernal_size * chanels + bias) * filters 
# 1번째 레이어 : (2 * 2 * 1 + 1) * 9 = 45
# 2번째 레이어 : (3 * 3 * 9 + 1) * 10 = 820
# 3번쨰 레이어 : (4* 4 * 10 + 1) * 15 = 2415
# Flattten 레이어 : reshape만 할 뿐 연산은 0


#3.  컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['acc'])
model.fit(X_train, y_train, batch_size=32, verbose=1, epochs=100, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(X_test, y_test)
print(results)
print('loss : ' , results[0])
print('acc : ' , results[1])

