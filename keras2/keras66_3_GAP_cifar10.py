import numpy as np
from keras.datasets import cifar10
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.preprocessing import OneHotEncoder

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape, y_train.shape) #   (50000, 32, 32, 3) (50000, 1)
print(X_test.shape, y_test.shape)   #   (10000, 32, 32, 3) (10000, 1)

# print(np.unique(y_train), np.unique(y_test))

X_train = X_train / 255.
X_test = X_test / 255.

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

#2
model = Sequential()
model.add(Conv2D(100, (2,2) , strides=1
                 , padding='same' # default : valid (2,2)
                 , input_shape=(32,32,3)) )
model.add(MaxPooling2D())
model.add(Conv2D(filters=100, kernel_size=(2,2), activation='swish')) 
model.add(Conv2D(100, (2,2), activation='swish'))           # (3, 3, 100)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())     # (100,)
model.add(Dense(units=50))              
model.add(Dense(10, activation='softmax'))

model.summary()

#3.  컴파일, 훈련

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['acc'])
model.fit(X_train, y_train, batch_size=32, verbose=1, epochs=100, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(X_test, y_test)
print(results)
print('loss : ' , results[0])
print('acc : ' , results[1])

