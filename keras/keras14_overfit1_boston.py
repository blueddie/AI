#09_1 카피

from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import r2_score, mean_squared_error
warnings.filterwarnings('ignore') # 워닝 무시
import time
import numpy as np
import pandas as pd



# 현재 사이킷런 버전 1.3.0 보스턴 안됨. 따라서 삭제
# pip uninstall scikit-learn
# pip uninstall scikit-learn-intelex
# pip uninstall scikit-image

# pip install scikit-learn==0.23.2  => 0.23.2 버전 설치
datasets = load_boston()
# print(datasets)
# x = datasets.data
# y = datasets.target
# print(x)  
# print(y)
# print(x.shape)  #(506, 13)
# print(y.shape)  #(506,)

# print(datasets.feature_names)
#   ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#       'B' 'LSTAT']

# print(datasets.DESCR) : 컬럼 설명

# [실습]
# train_size 0.7이상, 0.9이하
# R2 0.8 이상

#1. 데이터
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.7)

#2 모델 구성
model = Sequential()
model.add(Dense(8, input_dim=13))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#.컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
start_time = time.time()

hist = model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.2)
end_time = time.time()
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
results = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("R2 score : ", r2)
print("loss : " , loss)

def RMSE(aaa, bbb):
    return np.sqrt(mean_squared_error(aaa, bbb))
    
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
print("소요 시간 : ", end_time - start_time)
print("======================== hist ===========================================")
print(hist) # wrapping 된 상태
print("========================== hist.history =========================================")
print(hist.history)
print("========================= loss ==============================")
print(hist.history['loss'])
print("========================= val_loss ==============================")
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss', marker='.')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')
plt.legend(loc='upper right')
plt.title('boston loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()  
plt.show()


