from keras.datasets import cifar10
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from sklearn.preprocessing import OneHotEncoder
import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# print(X_train.shape)    #(50000, 32, 32, 3)
# print(X_test.shape)     #(10000, 32, 32, 3)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


print(X_train.shape)    #((50000, 3072)
print(X_test.shape)     #(10000, 3072)

X_train = np.asarray(X_train).astype(np.float32) 
X_test = np.asarray(X_test).astype(np.float32)