# https://www.kaggle.com/competitions/bike-sharing-demand/leaderboard
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
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
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

def RMSLE(y_test, y_predict):
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_predict))
    
    return rmsle


# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=7)
# model.compile(loss='mse', optimizer='adam')

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=90
                   , verbose=1
                   , restore_best_weights=True
                   )

# hist = model.fit(X_train, y_train, epochs=80, batch_size=150
#                  , validation_split=0.2
#                  ,callbacks=[es])

# loss = model.evaluate(X_test, y_test)

# y_predict = model.predict(X_test)

# r2 = r2_score(y_test, y_predict)

# y_submit = model.predict([test_csv])

# submission_csv['count'] = y_submit

# random_state=random_state
def auto() :
    random_state = random.randrange(1, 999999999)
    batch_size = random.randrange(32, 257)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,  stratify=X['season'], random_state=random_state)
    model.compile(loss='mse', optimizer='adam')

    hist = model.fit(X_train, y_train, epochs=900, batch_size=batch_size
                     , validation_split=0.2 
                     , callbacks=[es]
                     )
    
    y_predict = model.predict(X_test)
    
    
    rmsle = RMSLE(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    y_submit = model.predict([test_csv])
    submission_csv['count'] = y_submit
    print(rmsle)
    return rmsle, r2, random_state , batch_size, hist

min_rmsle = 1.35
max_r2 = 0.6
date = "0110_"
extension_name = ".csv"

# rmse, r2, rs, bs, hist = auto()
# print(rmse)
# print(r2)
# print(rs)
# print(bs)


while True :
    
    rmsle, r2, rs, bs, hist = auto()
    
    if rmsle < min_rmsle:
        min_rmsle = rmsle
        submission_csv.to_csv(path + date + str(rs) +"and"+ str(bs) + "and"+ str(round(rmsle, 2)) + extension_name, index=False)
        
        
# str(rs)




#  submission_csv.to_csv(path + date + str(batch_size) + "and" + str(random_state) +"andRMSE_" + str(rmse) + last_name , index=False)

# print("R2 score : ", r2)
# print("loss : " , loss)

# print("========================== hist.history =========================================")
# print(hist.history)
# print("========================= loss ==============================")
# print(hist.history['loss'])
# print("========================= val_loss ==============================")
# print(hist.history['val_loss'])

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'   #한글 깨짐 해결






# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss', marker='.')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')
# plt.legend(loc='upper right')
# plt.title('캐글 bike loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()  
# plt.show()

