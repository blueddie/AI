import numpy as np
import time
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D, GlobalAveragePooling2D, Input
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime


#1
np_path = 'c:/_data/_save_npy/'

x = np.load(np_path + 'keras39_5_men_women_x_np.npy')
y = np.load(np_path + 'keras39_5_men_women_y_np.npy')
test = np.load(np_path + 'keras39_5_men_women_test_np.npy')

print(x.shape, y.shape)
