from keras.datasets import cifar10
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.preprocessing import OneHotEncoder

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(np.unique(y_train, return_counts=True))

y_train = OneHotEncoder(sparse=False).fit_transform(y_train)
y_test = OneHotEncoder(sparse=False).fit_transform(y_test)

model = Sequential()
model.add(Conv2D(30, (2,2), input_shape=(32, 32, 3)))
model.add(Conv2D(15, (3,3)))
model.add(Conv2D(30, (2,2)))
model.add(Flatten())
model.add(Dense(40))
model.add(Dropout(0.3))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10, activation='softmax'))

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='auto', patience=200, verbose=1, restore_best_weights=True)
#3.  컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['acc'])

import time
st = time.time()
model.fit(X_train, y_train, batch_size=2000, verbose=1, epochs=10000, validation_split=0.2)
et = time.time()

#4. 평가, 예측
results = model.evaluate(X_test, y_test)
print(results)
print('loss : ' , results[0])
print('acc : ' , results[1])
print("걸린 시간 : ", et - st)

# acc 0.77 이상