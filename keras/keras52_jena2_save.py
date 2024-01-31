# 5일분 720행을 훈련시켜서
# 하루 144행 뒤를 예측
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import random, datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

csv_path = "C:\\_data\\kaggle\\jena\\"

train_csv = pd.read_csv(csv_path + "jena_climate_2009_2016.csv", index_col=0)

target_column_name = 'T (degC)'
moved_column_df = train_csv.pop(target_column_name)
train_csv[target_column_name] = moved_column_df
print('변경 전',train_csv.shape)    #변경 전 (420551, 14)

size = 720

def split_xy(dataset, size, column):
    x_arr, y_arr = [], []
    
    for i in range(dataset.shape[0] - size -144) :
        x_subset = dataset.iloc[i : (i + size), :]
        y_subset = dataset.iloc[(i + size + 144) , column]
        x_arr.append(x_subset)
        y_arr.append(y_subset)
    return np.array(x_arr), np.array(y_arr)
    
# def split_xy(data, time_step, y_col,y_gap=0):
#     result_x = []
#     result_y = []
    
#     num = len(data) - (time_step+y_gap)                 # x만자른다면 len(data)-time_step+1이지만 y도 잘라줘야하므로 +1이 없어야함
#     for i in range(num):
#         result_x.append(data[i : i+time_step])  # i 부터  time_step 개수 만큼 잘라서 result_x에 추가
#         y_row = data.iloc[i+time_step]          # i+time_step번째 행, 즉 result_x에 추가해준 바로 다음순번 행
#         result_y.append(y_row[y_col])           # i+time_step번째 행에서 원하는 열의 값만 result_y에 추가
    
#     return np.array(result_x), np.array(result_y)
   
    
            
x, y = split_xy(train_csv, size, 13)
       
# print('x: ' , x)
# print('y : ' , y)       
       
print('변경 후', x.shape)   #

np_path = 'c:\\_data\\_save_npy\\'
np.save(np_path + 'kaggle_jena_x720.npy', arr=x)
np.save(np_path + 'kaggle_jena_y720.npy', arr=y)