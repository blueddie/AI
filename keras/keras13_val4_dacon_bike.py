# https://dacon.io/competitions/open/235576/codeshare/6969?page=1&dtype=recent // 대회 주소

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import random
import time

#1. 데이터
path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path + "submission.csv")   #, index_col=0
print(submission_csv)

# print(train_csv.shape)  #(1459, 10)
# print(test_csv.shape)   #(715, 9)
# print(submission_csv.shape) #(715, 2)

# print(train_csv.columns)
# # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
# #        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
# #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'
# print(train_csv.info())
# print(test_csv.info())

# print(train_csv.info()) #std 표준편차, 50% 중위값
# print(test_csv.info())


#################결측치 제거###################
# print(train_csv.isnull().sum())
# print(train_csv.isna().sum())   #위 코드와 같다.
# train_csv = train_csv.dropna()  # 한 행에 결측기가 하나라도 있다면 그 행 전체를 삭제!!!
# print(train_csv.isna().sum())   #위 코드와 같다.
# print(train_csv.info())
# print(train_csv.shape)  #(1328, 10)

# train_csv = train_csv.fillna(train_csv.mean())            
# test_csv = test_csv.fillna(test_csv.mean())


#---------------------------------------------------------------------------위2개각 day01 결측치 처리


X = train_csv.drop(['count', 'hour_bef_ozone', 'hour_bef_pm2.5','hour_bef_pm10'], axis=1)

y = train_csv["count"]

test_csv = test_csv.drop(['hour_bef_ozone', 'hour_bef_pm2.5','hour_bef_pm10'], axis=1)
test_csv = test_csv.fillna(test_csv.mean())


X['hour_bef_temperature'] = X['hour_bef_temperature'].fillna(X['hour_bef_temperature'].mean())



# df[df.column.isna( )
# df[df['A'].isnull()]

# print(X[X['hour_bef_humidity'].isnull()]) #결측치가 있는 행 id 확인


# print(X[X['hour_bef_precipitation', 1].fillna()])
# subway_df.loc[165]

X.loc[[1420],['hour_bef_precipitation']] = 0.0
X.loc[[1553],['hour_bef_precipitation']] = 1.0

# print("결측 행은", X['hour_bef_precipitation'][X['hour_bef_precipitation'].isnull()])


# print(X.isnull().sum())

# X_sanple = X[X['hour'] == 18]
# print(X_sanple.head(10)['hour_bef_humidity'], X_sanple.head(10)['hour_bef_precipitation'])

X.loc[[1420],['hour_bef_humidity']] = 37.0
X.loc[[1553],['hour_bef_humidity']] = 82.0


# print(X[X['hour_bef_visibility'].isnull()])
X['hour_bef_visibility'] = X['hour_bef_visibility'].fillna(X['hour_bef_visibility'].mean())
X['hour_bef_windspeed'] = X['hour_bef_windspeed'].fillna(X['hour_bef_windspeed'].mean())
# X['hour_bef_pm10'] = X['hour_bef_pm10'].fillna(X['hour_bef_pm10'].mean())



print(test_csv.info())

first_name = "submission_0109_2"
second_name = ".csv"
i = 1
haha = 0

#2. 모델
model = Sequential()
model.add(Dense(8, input_dim=6, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))



########## x , y 분리 #############



print(y)


def auto_jjang():
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=53145880)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    #3
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=1000, batch_size=32
              , validation_split=0.2)
    #4. 평가,예측
    loss = model.evaluate(X_test, y_test)
    y_predict = model.predict(X_test)
    r2 = r2_score(y_test, y_predict)

    y_submit = model.predict([test_csv])
    print("loss : ",loss)
    print("r2 : ", r2)
    # print(y_submit)
    # print(y_submit.shape)


    ############### submission.csv 만들기 (count column에 값만 넣어주면 됨) ################
    submission_csv['count'] = y_submit

    return loss, r2
#---------------------
max_r2 = 0.6

while True :
    loss, r2 = auto_jjang()
    if (r2 > max_r2) :
        max_r2 = r2
        submission_csv.to_csv(path  + first_name + str(r2) + second_name, index=False)
        
    




# max_r2 = 0.60
# while haha <= 100 :
#     i = random.randrange(1,99999999)
#     r = random.randrange(16,65)
#     loss, r2 = auto_jjang(i, r)
    
#     if r2 > max_r2 :
          
#         bs = r  
#         rs = i
#         submission_csv.to_csv(path + first_name + str(bs)+ "and" + str(rs) +"andR2" + str(r2) + second_name, index=False)
#         haha = haha + 1
#         i = i + 1
#         max_r2 = r2 + 0.01
#         min_loss = loss - 1
#     else :
#         i = i + 1
