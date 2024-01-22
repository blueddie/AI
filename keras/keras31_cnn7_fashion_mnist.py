from keras.datasets import fashion_mnist
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print(X_train.shape)     # (60000, 28, 28)
print(y_train.shape)    # (60000,)
print(X_test.shape)    #(10000, 28, 28)
print(y_test.shape)   # (10000,)

# import matplotlib.pyplot as plt
# plt.imshow(X_train[15155], 'gray')
# plt.show()
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)


X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

scaler = MinMaxScaler()
scaler.fit(X_train_flattened)
scaled_train = scaler.transform(X_train_flattened)
scaled_test = scaler.transform(X_test_flattened)

X_train = scaled_train.reshape(X_train.shape)
X_test = scaled_test.reshape(X_test.shape)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


ohe = OneHotEncoder(sparse=False)
ohe.fit(y_train)
y_train = ohe.transform(y_train)
y_test = ohe.transform(y_test)


# model = Sequential()
# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)))
# model.add(Conv2D(128, (3,3), activation='swish'))
# model.add(Conv2D(57, (3,3), activation='swish'))
# model.add(Conv2D(31, (3,3), activation='relu'))
# model.add(Conv2D(15, (3,3), activation='swish'))
# model.add(Conv2D(17, (3,3), activation='relu'))
# model.add(GlobalMaxPooling2D())
# model.add(Dense(63, activation='swish'))
# model.add(Dense(10, activation='softmax'))
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
es = EarlyStopping(monitor='val_accuracy', mode='auto', patience=30, verbose=0, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])

import time
st = time.time()
model.fit(X_train, y_train, batch_size=16, verbose=1, epochs=120, validation_split=0.25, callbacks=[es])
et = time.time()

#4. 평가, 예측
results = model.evaluate(X_test, y_test)
print(results)
print('loss : ' , results[0])
print('acc : ' , results[1])
print("걸린 시간 : ", et - st)
