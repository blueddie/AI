import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error

#1. 데이터
# X = np.array(1,17)
X = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
# y = np.array(1,17)
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

## 잘라라
# 1- 10train 11 12 13validation 14, 15, 16 test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, shuffle=False)

# #2
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
model.fit(X_train, y_train, epochs=100, batch_size=312, validation_data=(X_val, y_val))

# #4
loss = model.evaluate(X_test, y_test)
results = model.predict([1313])
print("loss : ", loss)
print("예측값은 : ", results)