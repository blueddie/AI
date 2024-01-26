from keras.datasets import cifar10
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from sklearn.preprocessing import OneHotEncoder
import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   
path = '..\\_data\_save\\MCP\\cifar\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'cifar_', date, '_' ,filename])

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape)    #(50000, 32, 32, 3)
# print(y_train.shape)    #(50000, 1)

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

augument_size = 20000

randidx = np.random.randint(x_train.shape[0], size=augument_size)

x_augumented = x_train[randidx].copy() 
y_augumented = y_train[randidx].copy()

# print(x_augumented.shape)   #(20000, 32, 32, 3)
# print(y_augumented.shape)   #(20000, 32, 32, 3)

x_augumented = train_datagen.flow(
    x_augumented, y_augumented
    , batch_size=augument_size
    , shuffle=False
    ).next()[0]

# print(x_augumented.shape)   #(20000, 32, 32, 3)
# print(x_train.shape)    #(50000, 32, 32, 3)
# print(x_test.shape)     #(10000, 32, 32, 3)

x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))

print(x_train.shape, y_train.shape) #(70000, 32, 32, 3) (70000, 1)

ohe = OneHotEncoder(sparse=False)
ohe.fit(y_train)
y_train = ohe.transform(y_train)
y_test = ohe.transform(y_test)

#2
model = Sequential()
model.add(Conv2D(97, (3,3), activation='swish', input_shape=(32, 32, 3)))
model.add(Dropout(0.5))
model.add(Conv2D(160, (3,3), activation='swish', padding='same'))
model.add(Dropout(0.5))
model.add(Conv2D(120, (3,3), activation='swish', padding='same'))
model.add(GlobalMaxPooling2D())
model.add(Dense(50, activation='swish'))
model.add(Dense(30, activation='swish'))
model.add(Dense(10, activation='softmax'))


es = EarlyStopping(monitor='val_accuracy', mode='auto', patience=20, verbose=0, restore_best_weights=True)
#3.  컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])

import time
st = time.time()
model.fit(x_train, y_train, batch_size=64, verbose=1, epochs=110, validation_split=0.2, callbacks=[es])
et = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results)
print('loss : ' , results[0])
print('acc : ' , results[1])
print("걸린 시간 : ", et - st)

# 증폭 전
# loss :  0.7502949237823486
# acc :  0.744700014591217
# 걸린 시간 :  1061.7357642650604

# 증폭 후


