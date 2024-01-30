import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import random, datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

csv_path = "C:\\_data\\kaggle\\jena\\"

train_csv = pd.read_csv(csv_path + "jena_climate_2009_2016.csv", index_col=0)

# # print(train_csv.shape)  #(420551, 14)
target_column_name = 'T (degC)'
moved_column_df = train_csv.pop(target_column_name)
train_csv[target_column_name] = moved_column_df

print('변경 전',train_csv.shape)

size = 6

def split_x(dataset, size, column):
    x_arr = []
    y_arr = []
    for i in range(dataset.shape[0] - size) :
        x_subset = dataset.iloc[i : (i + size), :]
        y_subset = dataset.iloc[(i + size) , column]
        x_arr.append(x_subset)
        y_arr.append(y_subset)
        # print(subset)
    return np.array(x_arr), np.array(y_arr)

x, y = split_x(train_csv, size, 13)

# print('변경 후', x.shape)   #(420546, 6, 14)
# print(type(x)) #<class 'numpy.ndarray'>

print(y)

np_path = 'c:\\_data\\_save_npy\\'
np.save(np_path + 'kaggle_jena_x.npy', arr=x)
np.save(np_path + 'kaggle_jena_y.npy', arr=y)
