import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(X_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

# X_train = X_train.reshape(60000, 28, 28, 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# print(X_train.shape)   # (60000, 784)
# print(X_test.shape)    #(10000, 784)
X_train = np.asarray(X_train).astype(np.float32) 
X_test = np.asarray(X_test).astype(np.float32)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


ohe = OneHotEncoder(sparse=False)
ohe.fit(y_train)
y_train = ohe.transform(y_train)
y_test = ohe.transform(y_test)

# print(y_test.shape)
#2
model = Sequential()
model.add(Dense(1097, activation='swish', input_shape=(784,)))
model.add(Dropout(0.5))
model.add(Dense(877, activation='swish'))
model.add(Dropout(0.5))
model.add(Dense(964, activation='swish'))
model.add(Dropout(0.5))
model.add(Dense(877, activation='swish'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='swish'))
model.add(Dense(10, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode='auto', patience=20, verbose=0, restore_best_weights=True)

#3
model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
import time
st = time.time()
model.fit(X_train, y_train, batch_size=64, verbose=1, epochs=110, validation_split=0.2, callbacks=[es])
et = time.time()

#4
results = model.evaluate(X_test, y_test)
print(results)
print('loss : ' , results[0])
print('acc : ' , results[1])
print("걸린 시간 : ", et - st)