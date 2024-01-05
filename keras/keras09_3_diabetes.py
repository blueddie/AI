from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt


#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x)
# print(y)
# print(x.shape)    (442, 10)
# print(y.shape)    (442,)

# print(datasets.feature_names)
# print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1226, train_size=0.9)
# print(x_train.shape)
# print(y_train.shape)

#2
model = Sequential()
model.add(Dense(8, input_dim=10))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3
model.compile(loss='mae', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=100)
end_time = time.time()

#4
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
results = model.predict(x)

print("R2 score : ", r2)
print("loss : " , loss)
print("소요 시간 : ", round(end_time - start_time, 2), "seconds")


# model.fit(x_train, y_train, epochs=1000, batch_size=10 random_state=12555, train_size=0.9)
# R2 score :  0.5990970562932332
# loss :  1912.0970458984375
# 소요 시간 :  21.317250967025757 seconds

# batch_size=15
# R2 score :  0.6065895860879443
# loss :  1876.361572265625
# 소요 시간 :  14.900413513183594 2 seconds


# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=713, train_size=0.85)
# model.fit(x_train, y_train, epochs=1600, batch_size=15)
# R2 score :  0.6131244833494858
# loss :  2305.59765625
# 소요 시간 :  31.48 seconds

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=713, train_size=0.85)
# model.fit(x_train, y_train, epochs=100, batch_size=2)
# R2 score :  0.6455998483624379
# loss :  39.64200210571289
# 소요 시간 :  10.01 seconds
 
# epo:100 bs:2 rs:713 ts:0.87 mae
# R2 score :  0.6561132973500444
# loss :  40.575645446777344
# 소요 시간 :  9.75 seconds


# epo:50 bs:3 rs:713 ts:0.87 mse
# R2 score :  0.6593895518208093
# loss :  2290.4697265625
# 소요 시간 :  3.61 seconds

# epo:50 bs:3 rs:1226 ts:0.9 mae
# R2 score :  0.7249858415964545
# loss :  36.09334182739258
# 소요 시간 :  3.1 seconds
