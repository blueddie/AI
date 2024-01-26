import numpy as np
import time
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D, GlobalAveragePooling2D, Input
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import os
from keras.preprocessing.image import ImageDataGenerator


#1
np_path = 'c:/_data/_save_npy/'

x = np.load(np_path + 'keras39_3_cat_dog_x_np.npy')
y = np.load(np_path + 'keras39_3_cat_dog_y_np.npy')
test = np.load(np_path + 'kaggle_cat_dog_submission_np.npy')

# print(x.shape, y.shape) #(19997, 120, 120, 3) (19997,)

#2
model = Sequential()
model.add(Conv2D(16, (2,2) , strides=2, input_shape=(120, 120, 3)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.4))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(units=64))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dense(1, activation='sigmoid'))
model.summary()


x_train, x_test, y_train, y_test = train_test_split(x, y ,random_state=1452, train_size=0.86, stratify=y)

# print(x_train.shape, x_test.shape)  #(17197, 120, 120, 3) (2800, 120, 120, 3)

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

augument_size = 8000

randidx = np.random.randint(x_train.shape[0], size=augument_size)

x_augumented = x_train[randidx].copy() 
y_augumented = y_train[randidx].copy()


