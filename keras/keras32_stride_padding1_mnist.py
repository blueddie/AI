import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.preprocessing import OneHotEncoder

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(X_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] ,X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

ohe = OneHotEncoder(sparse=False)
ohe.fit(y_train)
y_train = ohe.transform(y_train)
y_test = ohe.transform(y_test)

# print(y_train.shape)
#2
model = Sequential()
model.add(Conv2D(9, (3,3), input_shape=(28,28,1)))
model.add(Conv2D(filters=10, kernel_size=(3,3) , padding='same')) 
model.add(Conv2D(15, (3, 3), padding='same'))
model.add(Conv2D(7, (3,3)))
model.add(Flatten())
model.add(Dense(units=40))
model.add(Dense(30, input_shape=(40,)))
#                   shape=(batch_size, input_dim)
model.add(Dense(20))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['acc'])
model.fit(X_train, y_train, batch_size=32, verbose=1, epochs=100, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(X_test, y_test)
print(results)
print('loss : ' , results[0])
print('acc : ' , results[1])

# loss :  0.31085267663002014
# acc :  0.9178000092506409