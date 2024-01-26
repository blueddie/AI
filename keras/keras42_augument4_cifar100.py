from keras.datasets import cifar100
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from sklearn.preprocessing import OneHotEncoder
import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler


(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape, x_test.shape)     #(50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape)     #(50000, 1) (10000, 1)

x_train / 255.
x_test / 255.

# print(x_train.shape, x_test.shape)

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

augument_size = 30000

randidx = np.random.randint(x_train.shape[0], size=augument_size)

x_augumented = x_train[randidx].copy() 
y_augumented = y_train[randidx].copy()

# print(x_augumented.shape)   #(30000, 32, 32, 3)
# print(y_augumented.shape)   #(30000, 1)


x_augumented = train_datagen.flow(
    x_augumented, y_augumented
    , batch_size=augument_size
    , shuffle=False
    ).next()[0]

x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))

# print(x_train.shape, y_train.shape) #(80000, 32, 32, 3) (80000, 1)

ohe = OneHotEncoder(sparse=False)
ohe.fit(y_train)
y_train = ohe.transform(y_train)
y_test = ohe.transform(y_test)



#2
model = Sequential()
model.add(Conv2D(19, (3,3), activation='swish', padding='same' , input_shape=(32, 32, 3)))
# model.add(MaxPool2D((2, 2)))
model.add(Conv2D(97, (3,3), activation='swish'))
# model.add(MaxPool2D((2, 2)))
model.add(Conv2D(143, (3,3), activation='swish' , padding='same'))
model.add(Flatten())
model.add(Dense(12, activation='swish'))
# model.add(Dense(11, activation='swish'))
model.add(Dense(56, activation='swish'))
model.add(Dense(100, activation='softmax'))
model.summary()

#3
es = EarlyStopping(monitor='val_accuracy', mode='auto', patience=10, verbose=0, restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
import time
st = time.time()

model.fit(x_train, y_train, batch_size=512, verbose=1, epochs=200, validation_split=0.2, callbacks=[es])

et = time.time()

#4
results = model.evaluate(x_test, y_test)
print(results)
print('loss : ' , results[0])
print('acc : ' , results[1])
print("걸린 시간 : ", et - st)

# 증폭 전




# 증폭 후



         
                        
