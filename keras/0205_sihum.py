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
samsung1 = samsung.iloc[:1418, :]
# print(samsung1)   #(1418, 15)
# print(pd.isna(samsung1).sum())  # 결측치 없음
amore1 = amore.iloc[:1418, :]
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

timesteps = 20

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

x_samsung, y_samsung = split_x(samsung1, timesteps, 14)
# print(x_samsung.shape)  #(1397, 20, 15)
# print(y_samsung)  #((1397,)

x_amore, y_amore = split_x(amore1, timesteps, 14)
# print(x_amore.shape)    #(1397, 20, 15)
# print(y_amore.shape)    #(1397,)

# x_samsung_pred = samsung1.tail(timesteps)
# x_amore_pred = amore1.tail(timesteps)
# print(x_samsung_pred)


x_samsung_train, x_samsung_test, x_amore_train, x_amore_test, y_samsung_train, y_samsung_test, y_amore_train, y_amore_test = train_test_split(x_samsung, x_amore, y_samsung, y_amore, shuffle=False, train_size=0.8)

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

scaler1 = MinMaxScaler()
scaler1.fit(x_samsung_train)
x_samsung_train = scaler1.transform(x_samsung_train)
x_samsung_test = scaler1.transform(x_samsung_test)

scaler2 = MinMaxScaler()
scaler2.fit(x_amore_train)
x_amore_train = scaler2.transform(x_amore_train)
x_amore_test = scaler2.transform(x_amore_test)

x_samsung_train = x_samsung_train.reshape(x_samsung_train.shape[0], timesteps, 15)
x_samsung_test = x_samsung_test.reshape(x_samsung_test.shape[0], timesteps, 15)

x_amore_train = x_amore_train.reshape(x_amore_train.shape[0], timesteps, 15)
x_amore_test = x_amore_test.reshape(x_amore_test.shape[0], timesteps, 15)

#2-1 모델
input1 = Input(shape=(timesteps,15))
dense1 = Bidirectional(LSTM(64, return_sequences=True))(input1)
dense2 = Dense(32)(dense1)
dense3 = Dense(32)(dense2)
dense4 = Dense(32)(dense3)
output1 = Dense(16)(dense4)

#2-2 모델
input11 = Input(shape=(timesteps,15))
dense11 = Bidirectional(LSTM(64, return_sequences=True))(input11)
dense12 = Dense(32)(dense11)
dense13 = Dense(32)(dense12)
dense14 = Dense(32)(dense13)
output11 = Dense(16)(dense14)

#2-3 concatnate
merge1 = concatenate([output1, output11], name='mg1')
merge2 = Conv1D(32, kernel_size=3, name='mg2')(merge1)
merge3 = Dense(16, name='mg3')(merge2)
merge4 = Dense(8, name='mg4')(merge3)
last_output1 = Dense(1, name='last1')(merge4)
last_output2 = Dense(1, name='last2')(merge4)

model = Model(inputs=[input1, input11], outputs=[last_output1, last_output2])

model.summary()

#3
es = EarlyStopping(monitor='val_loss', mode='min', patience=200, verbose=1, restore_best_weights=True)

model.compile(loss='mae', optimizer='adam')
model.fit([x_samsung_train, x_amore_train], [y_samsung_train, y_amore_train], batch_size=5, epochs=1000, callbacks=[es], validation_split=0.2)

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

# test_predict = model.predict([x_samsung_pred, x_amore_pred])
# print(y_predict[0])
# for i in range(len(y_predict[0])):
#     print('삼성 실제 값: ', y_samsung_test[i], '/ 에측가 : ', y_predict[0][i])
# for i in range(len(y_predict[1])):
#     print('아모레 실제 값: ', y_amore_test[i], '/ 에측가 : ', y_predict[1][i])
# model.save_weights('C:\\_data\\sihum\\' + date + '_ss_' + str(round(r2, 3)) + '_am_'+ str(round(r22, 3)) + 'save_weights.h5')
# print('삼성 시가 : ' , test_predict[0])
# print('아모레 종가 : ', test_predict[1])


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
