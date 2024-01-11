# https://www.kaggle.com/competitions/bike-sharing-demand/leaderboard
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import random
import time

#1. 데이터
path = "C://_data//kaggle//bike//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")

X = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']
test_csv = test_csv.drop([], axis=1)

#2. 모델
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

def RMSLE(y_test, y_predict):
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_predict))
    
    return rmsle

def ACC(aaa, bbb):
    return accuracy_score(aaa,bbb)


from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=50
                   , verbose=1
                   , restore_best_weights=True
                   )

def auto() :
    random_state = random.randrange(1, 999999999)
    batch_size = random.randrange(128, 513)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random_state)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    hist = model.fit(X_train, y_train, epochs=1200, batch_size=batch_size
                     , validation_split=0.2 
                     , callbacks=[es]
                     )
    
    y_predict = model.predict(X_test)
    
    loss = model.evaluate(X_test, y_test)
    rmsle = RMSLE(y_test, y_predict)
    # r2 = r2_score(y_test, y_predict)
    y_submit = model.predict([test_csv])
    submission_csv['count'] = y_submit
    # print("rmsle", rmsle)
    # print("loss", loss[0])
    
    return rmsle, random_state , batch_size, loss[0]

min_rmsle = 1.3
min_loss = 999999
date = "0111_3_"
extension_name = ".csv"

# rmsle, rs, bs, loss, hist = auto()
while True :
    rmsle, rs, bs, loss = auto()
    
    if rmsle < min_rmsle and loss < min_loss:
        
        min_loss = loss
        min_rmsle = rmsle
        submission_csv.to_csv(path + date + str(rs) +"and"+ str(bs) + "and"+ str(rmsle) + extension_name, index=False)
        
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss', marker='.')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')
# plt.legend(loc='upper right')
# plt.title('kaggle_bike_loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()  
# plt.show()
        