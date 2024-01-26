from keras.datasets import fashion_mnist
from keras.models import Sequential, Model
import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, MaxPooling2D, Input
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,StandardScaler, RobustScaler
import time
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train / 255.
x_test = x_test / 255.

train_datagen = ImageDataGenerator(
    # rescale=1./255
    horizontal_flip=True
    , vertical_flip=True
    , width_shift_range=0.2
    , height_shift_range=0.2
    , rotation_range=30
    , zoom_range=0.2
    , shear_range=20
    , fill_mode='nearest' # default: nearest
)

augument_size = 40000

randidx = np.random.randint(x_train.shape[0], size=augument_size)

x_augumented = x_train[randidx].copy()   # .copy() -> 별도의 메모리 공간을 할당해서 ?
y_augumented = y_train[randidx].copy()

# print(x_augmented)
# print(x_augmented.shape)    # (40000, 28, 28)
# print(y_augmented)
# print(y_augmented.shape)    # (40000,)

x_augumented = x_augumented.reshape(40000, 28, 28, 1)

x_augumented = train_datagen.flow(
    x_augumented, y_augumented
    , batch_size=augument_size
    , shuffle=False
    , save_to_dir='C:\\_data\\temp\\'
    ).next()[0]

# print(x_augmented.shape)    #(40000, 28, 28, 1)
# print(x_augmented[1].shape)    #(40000,)

# print(x_train.shape)    #(60000, 28, 28)
# print(x_test.shape)
'''
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))

print(x_train.shape, y_train.shape) #(100000, 28, 28, 1) (100000,)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

ohe = OneHotEncoder(sparse=False)
ohe.fit(y_train)
y_train = ohe.transform(y_train)
y_test = ohe.transform(y_test)

# print(y_train.shape)    #(100000, 10)
# print(y_test.shape)     #(10000, 10)

#2
model = Sequential()                    
model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu')) 
model.add(Dropout(0.5))
model.add(Conv2D(77, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(77, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(146, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(86, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(124, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

#3
strat_time = time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=200, batch_size=1024, verbose=1, validation_split=0.3, callbacks=[es])

end_time = time.time()

results = model.evaluate(x_test, y_test)
loss = results[0]
acc = results[1]

print('loss :' , loss)
print('acc : ' , acc)
print('걸린 시간' , round(end_time - strat_time, 2) , '초')


# loss : 0.2533127963542938
# acc :  0.9147999882698059
# 걸린 시간 81.97 초

# loss : 0.24786163866519928
# acc :  0.9147999882698059
# 걸린 시간 78.54 초

# batch 512
# loss : 0.25013139843940735
# acc :  0.9161999821662903
# 걸린 시간 75.86 초

#batch 256
# loss : 0.24970273673534393
# acc :  0.9162999987602234
# 걸린 시간 109.63 초
'''