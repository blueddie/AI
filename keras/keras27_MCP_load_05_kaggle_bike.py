# https://www.kaggle.com/competitions/bike-sharing-demand/leaderboard
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import random

#1. 데이터
path = "C://_data//kaggle//bike//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")

X = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']
test_csv = test_csv.drop([], axis=1)

#2. 모델

def RMSLE(y_test, y_predict):
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_predict))
    
    return rmsle

def ACC(aaa, bbb):
    return accuracy_score(aaa,bbb)



def auto(test_csv) :
    # random_state = random.randrange(1, 999999999)
    # batch_size = random.randrange(128, 513)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=15)
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)
    test_csv = scaler.transform(test_csv)
    
    model = load_model('..\\_data\_save\\MCP\\keras26_kaggle_bike.hdf5')

                     

    y_predict = model.predict(X_test)
    
    loss = model.evaluate(X_test, y_test)
    rmsle = RMSLE(y_test, y_predict)
    # r2 = r2_score(y_test, y_predict)
    y_submit = model.predict([test_csv])
    submission_csv['count'] = y_submit
    # print("rmsle", rmsle)
    # print("loss", loss[0])
    
    return rmsle, loss[0]

rmsle, loss = auto(test_csv)
print('rmsle : ' , rmsle)
print('loss : ' , loss)

# rmsle :  1.548550820622674
# loss :  31638.96484375
