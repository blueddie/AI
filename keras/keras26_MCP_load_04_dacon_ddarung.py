# https://dacon.io/competitions/open/235576/codeshare/6969?page=1&dtype=recent // 대회 주소
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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


def RMSE(y_test, y_predict):
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    
    return rmse
#2. 모델
def auto_jjang(test_csv):

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1226)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)
    test_csv = scaler.transform(test_csv)
    
    #3
    model = load_model('..\\_data\_save\\MCP\\keras26_ddarung.hdf5')
    
    #4. 평가,예측
    loss = model.evaluate(X_test, y_test)
    y_predict = model.predict(X_test)
    r2 = r2_score(y_test, y_predict)
    rmse = RMSE(y_test, y_predict)
    
    y_submit = model.predict([test_csv])

    submission_csv['count'] = y_submit
    if rmse < 0.80 :
        
        submission_csv.to_csv(path + "0116_minmax" + str(round(rmse, 3)) + ".csv", index=False)
    return loss[0], rmse

loss, rmse = auto_jjang(test_csv)
print("loss : ", loss)
print("rmse : " , rmse)

# loss :  6923.4462890625
# rmse :  83.20724884884876