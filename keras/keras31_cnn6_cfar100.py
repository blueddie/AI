from keras.datasets import cifar100
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPool2D
from sklearn.preprocessing import OneHotEncoder
import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   
path = '..\\_data\_save\\MCP\\cifar100\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'cifar_', date, '_' ,filename])

(X_train, y_train), (X_test, y_test) = cifar100.load_data()

# print(X_train.shape)    #(50000, 32, 32, 3)
# print(y_train.shape)    #((50000, 1)
# print(X_test.shape)    #(10000, 32, 32, 3)
# print(y_test.shape)    #(10000, 1)

X_train_flattened = X_train.reshape(X_train.shape[0], -1)
# print(X_train_flattened.shape)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

scaler = StandardScaler()
scaler.fit(X_train_flattened)
scaled_train = scaler.transform(X_train_flattened)
scaled_test = scaler.transform(X_test_flattened)

X_train = scaled_train.reshape(X_train.shape)
X_test = scaled_test.reshape(X_test.shape)

ohe = OneHotEncoder(sparse=False)
ohe.fit(y_train)
y_train = ohe.transform(y_train)
y_test = ohe.transform(y_test)

# print(y_train.shape)    #(50000, 100)
# print(y_test.shape)     #(10000, 100)


#2 
model = Sequential()
model.add(Conv2D(19, (4,4), activation='swish', input_shape=(32, 32, 3)))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(97, (3,3), activation='swish'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(143, (3,3), activation='swish'))
model.add(GlobalAveragePooling2D())
model.add(Dense(12, activation='swish'))
# model.add(Dense(11, activation='swish'))
model.add(Dense(56, activation='swish'))
model.add(Dense(100, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode='auto', patience=20, verbose=0, restore_best_weights=True)

#3
model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
import time
st = time.time()

model.fit(X_train, y_train, batch_size=1024, verbose=1, epochs=200, validation_split=0.2, callbacks=[es])

et = time.time()

#4
results = model.evaluate(X_test, y_test)
print(results)
print('loss : ' , results[0])
print('acc : ' , results[1])
print("걸린 시간 : ", et - st)


# GlobalAveragePooling2D
# loss :  2.2616653442382812
# acc :  0.4578000009059906
# 걸린 시간 :  1149.6282968521118