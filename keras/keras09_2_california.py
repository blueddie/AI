from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import r2_score
import time

#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x)
# print(y)
# print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=15, train_size=0.9)

#2.
model = Sequential()
model.add(Dense(8,input_dim=8))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3.
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=5000, batch_size=400)
end_time = time.time()

#4.
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
results = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("R2 score : ", r2)
print("loss : " , loss)
print("소요 시간 : ", round(end_time - start_time, 2), "seconds")

#[0.55~ 0.6]

# batch_size=400 epochs=1000 train_size=0.9 random_state=3
# R2 score :  0.5713381289176827
# loss :  0.577601432800293
# 소요 시간 :  27.01 seconds

# batch_size=400 epochs=3000 train_size=0.9 random_state=59
# R2 score :  0.5940393883786245
# loss :  0.5329024195671082
# 소요 시간 :  79.83 seconds

# batch_size=400 epochs=3000 train_size=0.9 random_state=3
# R2 score :  0.6091549966118657
# loss :  0.5266450047492981
# 소요 시간 :  79.78 seconds

# batch_size=400 epochs=5000 train_size=0.9 random_state=3
# R2 score :  0.6138764207125262
# loss :  0.520283043384552
# 소요 시간 :  132.46 seconds

# batch_size=400 epochs=6000 train_size=0.9 random_state=3
# R2 score :  0.6272705655420208
# loss :  0.5022349953651428
# 소요 시간 :  242.36 seconds

# batch_size=600 epochs=7000 train_size=0.9 random_state=3
# R2 score :  0.6233230460943255
# loss :  0.5075541734695435
# 소요 시간 :  130.5 seconds