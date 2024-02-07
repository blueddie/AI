from keras.models import Sequential
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate, LSTM, GRU, Bidirectional, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

def split_x(dataset, timesteps, column):
    x_arr = []
    y_arr = []
    for i in range(len(dataset) - timesteps - 1) :
        x_subset = dataset.iloc[i : (i + timesteps), :]
        y_subset = dataset.iloc[(i + timesteps + 1) , column]
        x_arr.append(x_subset)
        y_arr.append(y_subset)
        # print(subset)
    return np.array(x_arr), np.array(y_arr)

##################################################################################
csv_path = "C:\\_data\\sihum\\"

samsung = pd.read_csv(csv_path + '삼성 240205.csv', thousands=',', index_col=0, encoding='EUC-KR')
amore = pd.read_csv(csv_path + '아모레 240205.csv', thousands=',', index_col=0, encoding='EUC-KR')

new_samsung = samsung.copy()
new_amore = amore.copy()

target_column_name = '시가'
other_columns = new_samsung.drop(columns=[target_column_name])
new_samsung = pd.concat([other_columns, new_samsung[target_column_name]], axis=1)
# print(new_samsung)

target_column_name = '종가'
other_columns = new_amore.drop(columns=[target_column_name])
new_amore = pd.concat([other_columns, new_amore[target_column_name]], axis=1)

new_samsung = new_samsung.drop(['전일비'], axis=1)  # 우선 전알비 컬럼 제거
new_samsung = new_samsung.rename(columns={'Unnamed: 6' : '전일비'}) 

new_amore = new_amore.drop(['전일비'], axis=1)  # 우선 전알비 컬럼 제거
new_amore = new_amore.rename(columns={'Unnamed: 6' : '전일비'})

new_samsung = new_samsung.astype('float32')
new_amore = new_amore.astype('float32')

new_samsung = new_samsung.sort_values(['일자'], ascending=[True])
new_amore = new_amore.sort_values(['일자'], ascending=[True])
# print(new_samsung.shape)    #(10296, 15)
# print(new_amore.shape)      #(4350, 15)

# print(new_samsung.dtypes)
# print(new_amore.dtypes)
timesteps = 20
x_samsung, y_samsung = split_x(new_samsung, timesteps, 14)
x_amore, y_amore = split_x(new_amore, timesteps, 14)

print(x_samsung.shape, y_samsung.shape)
