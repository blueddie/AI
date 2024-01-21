import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.preprocessing import OneHotEncoder

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(X_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

# X_train = X_train.reshape(60000, 28, 28, 1)
# X_test = X_test.reshape(10000, 28, 28, 1)

#-------
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# print(X_test.shape)

print(y_train.shape)
# print(y_test.shape)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print(np.unique(y_train, return_counts=True))




y_train = OneHotEncoder(sparse=False).fit_transform(y_train)
y_test = OneHotEncoder(sparse=False).fit_transform(y_test)

# print(y_train.shape)
#2
model = Sequential()
model.add(Conv2D(9, (2,2) 
                 , input_shape=(28,28,1))
                    #input_shape = (batch_size, row, columns, chanels)
                    #input_shape = (batch_size, height, width, chanels)
          )
model.add(Conv2D(filters=10, kernel_size=(2,2))) 
model.add(Conv2D(15, (2,2)))
model.add(Conv2D(7, (2,2)))
model.add(Flatten())
model.add(Dense(units=40))
model.add(Dense(30, input_shape=(40,)))
#                   shape=(batch_size, input_dim)
model.add(Dense(20))
model.add(Dense(10, activation='softmax'))

model.summary()



'''
(kernal_size * kernal_size * chanels + bias) * filters 
1번째 레이어 : (2 * 2 * 1 + 1) * 9 = 45
2번째 레이어 : (3 * 3 * 9 + 1) * 10 = 820
3번쨰 레이어 : (4* 4 * 10 + 1) * 15 = 2415
Flattten 레이어 : reshape만 할 뿐 연산은 0
'''
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='auto', patience=30, verbose=1, restore_best_weights=True)
#3.  컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['acc'])

import time
st = time.time()
model.fit(X_train, y_train, batch_size=1000, verbose=1, epochs=10000, validation_split=0.2, callbacks=[es])
et = time.time()

#4. 평가, 예측
results = model.evaluate(X_test, y_test)
print(results)
print('loss : ' , results[0])
print('acc : ' , results[1])
print("걸린 시간 : ", et - st)