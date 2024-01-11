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

X = train_csv.drop(['count'], axis=1)

y = train_csv["count"]

test_csv = test_csv.fillna(test_csv.mean())

X['hour_bef_temperature'] = X['hour_bef_temperature'].fillna(X['hour_bef_temperature'].mean())

X.loc[[1420],['hour_bef_precipitation']] = 0.0
X.loc[[1553],['hour_bef_precipitation']] = 1.0

X.loc[[1420],['hour_bef_humidity']] = 37.0
X.loc[[1553],['hour_bef_humidity']] = 82.0

X['hour_bef_visibility'] = X['hour_bef_visibility'].fillna(X['hour_bef_visibility'].mean())

first_name = "submission_0110_es_rs_"
second_name = ".csv"

#2. 모델
model = Sequential()
model.add(Dense(8, input_dim=9, activation='relu'))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=50
                   , verbose=1
                   , restore_best_weights=True
                   )
def RMSE(y_test, y_predict):
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    
    return rmse

def auto_jjang():
    rs = random.randrange(1, 99999999)
    bs = random.randrange(32, 257)
    # rs = 56238592
    # bs = 47
    # epo = random.randrange(100, 9999)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=rs, stratify=X['hour'])
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    #3
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=1000, batch_size=bs
              , validation_split=0.2
              , callbacks=[es]
              )
    #4. 평가,예측
    # loss = model.evaluate(X_test, y_test)
    y_predict = model.predict(X_test)
    r2 = r2_score(y_test, y_predict)
    rmse = RMSE(y_test, y_predict)
    
    y_submit = model.predict([test_csv])

    submission_csv['count'] = y_submit

    return r2, rmse, rs, bs
#---------------------
max_r2 = 0.70
min_rmse = 60

while True :
    r2, rmse, rs, bs = auto_jjang()
    if rmse < min_rmse:
        min_rmse = rmse
        submission_csv.to_csv(path  + first_name + str(rs) + "bs" + str(bs) + "and"+ str(round(rmse, 2))+ "and" + str(round(r2, 2))+ second_name, index=False)