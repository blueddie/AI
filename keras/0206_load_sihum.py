from keras.models import Sequential
import pandas as pd
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, concatenate, LSTM, GRU, Bidirectional, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import tensorflow as tf
import random as rn
rn.seed(333)
tf.random.set_seed(123)     # 텐서 2.9 됨 2.15 안 먹음
np.random.seed(321)



csv_path = "C:\\_data\\sihum\\"

samsung = pd.read_csv(csv_path + '삼성 240205.csv', index_col=0, encoding='EUC-KR')
amore = pd.read_csv(csv_path + '아모레 240205.csv' , index_col=0, encoding='EUC-KR')

# print(samsung)  #[10296 rows x 16 columns]
# print(amore)    #[4350 rows x 16 columns]
# print(samsung.shape)    #(10296, 16)
# print(amore.shape)  #(4350, 16)
###############################################################################################################################################################
# samsung
# print(samsung.dtypes)
# 시가             object
# 고가             object
# 저가             object
# 종가             object
# 전일비            object
# Unnamed: 6     object
# 등락률           float64
# 거래량            object
# 금액(백만)         object
# 신용비           float64
# 개인             object
# 기관             object
# 외인(수량)         object
# 외국계            object
# 프로그램           object
# 외인비           float64
# dtype: object
##########################################
# amore
# print(amore.dtypes)
# 시가             object
# 고가             object
# 저가             object
# 종가             object
# 전일비            object
# Unnamed: 6     object
# 등락률           float64
# 거래량            object
# 금액(백만)         object
# 신용비           float64
# 개인             object
# 기관             object
# 외인(수량)         object
# 외국계            object
# 프로그램           object
# 외인비           float64
# dtype: object
################################################################################################################################################################
# print(pd.value_counts(samsung['전일비']))
#      3365
# ▲    3324
# ▼    3301
# ↑     210
# ↓      96
################################################################################################################################################################
samsung = samsung.drop(['전일비'], axis=1)  # 우선 전알비 컬럼 제거
samsung = samsung.rename(columns={'Unnamed: 6' : '전일비'}) # Unnamed: 6 -> 전일비로 바꿔주기

amore = amore.drop(['전일비'], axis=1)  # 우선 전알비 컬럼 제거
amore = amore.rename(columns={'Unnamed: 6' : '전일비'}) # Unnamed: 6 -> 전일비로 바꿔주기
# print(samsung)
# print(samsung.shape)    #(10296, 15)
# print(amore)
# print(amore.shape)    #(4350, 15)
################################################################################################################################################################
# str -> 숫자형
# '숫자열' 열에서 ',' 제거하고 float32로 변환
for col in samsung.columns:
    if samsung[col].dtype != 'float64':
        samsung[col] = pd.to_numeric(samsung[col].str.replace(',', ''), errors='coerce')
samsung = samsung.astype('float32')
# print(samsung.dtypes)
for col in amore.columns:
    if amore[col].dtype != 'float64':
        amore[col] = pd.to_numeric(amore[col].str.replace(',', ''), errors='coerce')
amore = amore.astype('float32')
# print(amore.dtypes)
################################################################################################################################################################
# 결측치 확인
# print(pd.isna(samsung).sum())
# 거래량            4   # 거래량 결측치 있음
# 금액(백만)        31  # 금액 결측치 있음

# print(pd.isna(amore).sum())
# 거래량       10      # 거래량 결측치 있음
# 금액(백만)    10     # 금액 결측치 있음
################################################################################################################################################################
# 결측치 행 확인
# print(samsung[pd.isna(samsung).any(axis=1)])
# print(amore[pd.isna(amore).any(axis=1)])
################################################################################################################################################################
samsung1 = samsung.iloc[1:1418, :]
# print(samsung1)   #(1418, 15)
# print(pd.isna(samsung1).sum())  # 결측치 없음
amore1 = amore.iloc[1:1418, :]
# print(amore1.shape) #(1418, 15)
# print(pd.isna(amore1).sum())    # 결측치 없음
################################################################################################################################################################
# # 데이터 분리
target_column_name = '시가'
other_columns = samsung1.drop(columns=[target_column_name])
samsung1 = pd.concat([other_columns, samsung1[target_column_name]], axis=1)
# print(samsung1)

target_column_name = '종가'
other_columns = amore1.drop(columns=[target_column_name])
amore1 = pd.concat([other_columns, amore1[target_column_name]], axis=1)
# print(amore1)

samsung1 = samsung1.sort_values(['일자'], ascending=[True])
amore1 = amore1.sort_values(['일자'], ascending=[True])

print(samsung1)

timesteps = 30


def split_x(dataset, timesteps, column):
    x_arr = []
    y_arr = []
    for i in range(len(dataset) - timesteps - 2) :
        x_subset = dataset.iloc[i : (i + timesteps), :]
        y_subset = dataset.iloc[(i + timesteps + 2) , column]
        x_arr.append(x_subset)
        y_arr.append(y_subset)
        # print(subset)
    return np.array(x_arr), np.array(y_arr)

x_samsung, y_samsung = split_x(samsung1, timesteps, 14)
# print(x_samsung.shape)  #(1397, 20, 15)
# print(y_samsung)  #((1397,)

x_amore, y_amore = split_x(amore1, timesteps, 14)
# print(x_amore.shape)    #(1397, 20, 15)
# print(y_amore.shape)    #(1397,)

x_samsung_pred = samsung1.tail(timesteps)
x_amore_pred = amore1.tail(timesteps)
x_samsung_pred = x_samsung_pred.values.reshape(-1, timesteps, 15)
x_amore_pred = x_amore_pred.values.reshape(-1, timesteps, 15)

x_samsung_train, x_samsung_test, x_amore_train, x_amore_test, y_samsung_train, y_samsung_test, y_amore_train, y_amore_test = train_test_split(x_samsung, x_amore, y_samsung, y_amore, shuffle=True,random_state=3, train_size=0.8)

# print(x_samsung_train.shape, x_samsung_test.shape)  #(1117, 20, 15) (280, 20, 15)
# print(y_samsung_train.shape, y_samsung_test.shape)  #(1117,) (280,)

# print(x_amore_train.shape, x_amore_test.shape)        #(1117, 20, 15) (280, 20, 15)
# print(y_amore_train.shape, y_amore_test.shape)        #(1117,) (280,)

x_samsung_train = x_samsung_train.reshape(x_samsung_train.shape[0],-1)
x_samsung_test = x_samsung_test.reshape(x_samsung_test.shape[0],-1)
# print(x_samsung_train.shape)    #(1117, 300)
# print(x_samsung_test.shape)     #(280, 300)
x_amore_train = x_amore_train.reshape(x_amore_train.shape[0],-1)
x_amore_test = x_amore_test.reshape(x_amore_test.shape[0],-1)

x_samsung_pred = x_samsung_pred.reshape(x_samsung_pred.shape[0],-1)
x_amore_pred = x_amore_pred.reshape(x_amore_pred.shape[0],-1)

scaler1 = MinMaxScaler()
scaler1.fit(x_samsung_train)
x_samsung_train = scaler1.transform(x_samsung_train)
x_samsung_test = scaler1.transform(x_samsung_test)

scaler2 = MinMaxScaler()
scaler2.fit(x_amore_train)
x_amore_train = scaler2.transform(x_amore_train)
x_amore_test = scaler2.transform(x_amore_test)

x_samsung_pred = scaler1.transform(x_samsung_pred)
x_amore_pred = scaler2.transform(x_amore_pred)

# x_samsung_pred = x_samsung_pred.reshape(x_samsung_pred.shape[0], timesteps, 15)
# x_amore_pred = x_amore_pred.reshape(x_amore_pred.shape[0], timesteps, 15)
# print(x_samsung_pred.shape)
# print(x_amore_pred.shape)

x_samsung_train = x_samsung_train.reshape(x_samsung_train.shape[0], timesteps, 15)
x_samsung_test = x_samsung_test.reshape(x_samsung_test.shape[0], timesteps, 15)

x_amore_train = x_amore_train.reshape(x_amore_train.shape[0], timesteps, 15)
x_amore_test = x_amore_test.reshape(x_amore_test.shape[0], timesteps, 15)


x_samsung_pred = x_samsung_pred.reshape(x_samsung_pred.shape[0], timesteps, 15)
x_amore_pred = x_amore_pred.reshape(x_amore_pred.shape[0], timesteps, 15)

#2-1 모델


model = load_model('c:\\_data\\sihum\\0206_1737_ss_0.985_am_0.965save_weights.hdf5')

#3

date = datetime.datetime.now().strftime("%m%d_%H%M")

#4
results = model.evaluate([x_samsung_test, x_amore_test],[y_samsung_test, y_amore_test])

y_predict = model.predict([x_samsung_test, x_amore_test])
# print(y_predict.shape)  #(280, 2)

r2 = r2_score(y_samsung_test, y_predict[0])
r22 = r2_score(y_amore_test, y_predict[1])

print('loss : ',results)
print('samsung r2 : ', r2)
print('amore r2 : ', r22)

test_predict = model.predict([x_samsung_pred, x_amore_pred])
# print(y_predict[0])
# for i in range(len(y_predict[0])):
#     print('삼성 실제 값: ', y_samsung_test[i], '/ 에측가 : ', y_predict[0][i])
# for i in range(len(y_predict[1])):
#     print('아모레 실제 값: ', y_amore_test[i], '/ 에측가 : ', y_predict[1][i])
# model.save('C:\\_data\\sihum\\' + date + '_ss_' + str(round(r2, 3)) + '_am_'+ str(round(r22, 3)) + 'save_weights.hdf5')
print('삼성 시가 : ' , np.round(test_predict[0], 2))
print('아모레 종가 : ', np.round(test_predict[1], 2))


# samsung r2 :  0.9294104626372455
# amore r2 :  0.08696188989836662

# samsung r2 :  0.8111561337868831
# amore r2 :  0.1571113451584636

# samsung r2 :  0.8955776195942577
# amore r2 :  -3.0508067064715387

# samsung r2 :  0.9204156521039752
# amore r2 :  0.5586651618650482
#######################################################

# samsung r2 :  0.9460253936001762
# amore r2 :  0.8414888743251627

# samsung r2 :  0.9492988137785269
# amore r2 :  0.8475916075595877
###################################################
#0206 
# samsung r2 :  0.9158839602781536
# amore r2 :  0.6646651316718233
# 삼성 시가 :  [[73022.78]]
# 아모레 종가 :  [[130798.1]]
###################################################

# samsung r2 :  0.8392551504303158
# amore r2 :  0.769789191759954
# 삼성 시가 :  [[77315.87]]
# 아모레 종가 :  [[123742.695]]
###################################################
# samsung r2 :  0.9858397323816692
# amore r2 :  0.9773879788640526
# 삼성 시가 :  [[72068.04]]
# 아모레 종가 :  [[116968.97]]
###################################################
# samsung r2 :  0.9883376639408199
# amore r2 :  0.9741992677102393
# 삼성 시가 :  [[74415.74]]
# 아모레 종가 :  [[118608.805]]
#################################################
# ts 30
