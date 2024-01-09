from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

#1. 데이터
X = np.array(range(1,17))
y = np.array(range(1,17))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.84, random_state=13)

#2
model = Sequential()
model.add(Dense(8, input_dim=1))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

# #3.
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=2
        #  ,validation_data=(X_val, y_val)
          , validation_split=0.3
          , verbose=1
          )

# #4
loss = model.evaluate(X_test, y_test)
results = model.predict([7])

y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
print("loss : ", loss)
print("예측값은 : ", results)
print("r2 :", r2)


# loss :  1.8189894035458565e-12
# 예측값은 :  [[7.0000024]]
# r2 : 0.9999999999993704