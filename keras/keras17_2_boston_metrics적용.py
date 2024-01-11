from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import r2_score, mean_squared_error
import time
import numpy as np
import pandas as pd

datasets = load_boston()


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
model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=10
                   , verbose=1
                   )


start_time = time.time()

hist = model.fit(x_train, y_train, epochs=200, batch_size=1
                 , validation_split=0.2
                 , callbacks=[es])
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