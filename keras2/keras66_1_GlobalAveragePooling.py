import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D


(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(X_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

print(X_test.shape)

#2
model = Sequential()
model.add(Conv2D(100, (2,2) , strides=1
                 , padding='same' # default : valid (2,2)
                 , input_shape=(10,10,1)) )
model.add(MaxPooling2D())
model.add(Conv2D(filters=100, kernel_size=(2,2))) 
model.add(Conv2D(100, (2,2)))           # (3, 3, 100)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())     # (100,)
model.add(Dense(units=50))              
model.add(Dense(10, activation='softmax'))

model.summary()

# GlobalAveragePooling
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 10, 10, 100)       500

#  max_pooling2d (MaxPooling2D  (None, 5, 5, 100)        0
#  )

#  conv2d_1 (Conv2D)           (None, 4, 4, 100)         40100

#  conv2d_2 (Conv2D)           (None, 3, 3, 100)         40100

#  global_average_pooling2d (G  (None, 100)              0
#  lobalAveragePooling2D)

#  dense (Dense)               (None, 50)                5050

#  dense_1 (Dense)             (None, 10)                510

# =================================================================
# Total params: 86,260
# Trainable params: 86,260
# Non-trainable params: 0


# Flatten
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 10, 10, 100)       500

#  max_pooling2d (MaxPooling2D  (None, 5, 5, 100)        0
#  )

#  conv2d_1 (Conv2D)           (None, 4, 4, 100)         40100

#  conv2d_2 (Conv2D)           (None, 3, 3, 100)         40100

#  flatten (Flatten)           (None, 900)               0

#  dense (Dense)               (None, 50)                45050

#  dense_1 (Dense)             (None, 10)                510

# =================================================================
# Total params: 126,260
# Trainable params: 126,260
# Non-trainable params: 0
_________________________________________________________________







# #3.  컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['acc'])
# model.fit(X_train, y_train, batch_size=32, verbose=1, epochs=100, validation_split=0.2)

# #4. 평가, 예측
# results = model.evaluate(X_test, y_test)
# print(results)
# print('loss : ' , results[0])
# print('acc : ' , results[1])
