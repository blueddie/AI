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
# print(train_csv)  [10886 rows x 11 columns]
test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv)   [6493 rows x 8 columns]
submission_csv = pd.read_csv(path + "sampleSubmission.csv")
# print(submission_csv) [6493 rows x 2 columns]


X = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']
# print(X.shape)   #(10886, 8)
# print(y.shape)  #(10886,)
# print(test_csv.shape) #(6493, 8)

#2. 모델
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

def RMSE(y_test, y_predict):
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    
    return rmse


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=3)
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
hist = model.fit(X_train, y_train, epochs=10, batch_size=42, validation_split=0.2)
end_time = time.time()
loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
y_submit = model.predict([test_csv])
submission_csv['count'] = y_submit
#  submission_csv.to_csv(path + date + str(batch_size) + "and" + str(random_state) +"andRMSE_" + str(rmse) + last_name , index=False)

print("R2 score : ", r2)
print("loss : " , loss)
print("소요 시간 : ", round(end_time - start_time, 2), "seconds")

print("========================== hist.history =========================================")
print(hist.history)
print("========================= loss ==============================")
print(hist.history['loss'])
print("========================= val_loss ==============================")
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   #한글 깨짐 해결


plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss', marker='.')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')
plt.legend(loc='upper right')
plt.title('캐글 bike loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()  
plt.show()