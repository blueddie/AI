from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터 
datasets = fetch_california_housing()
X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, train_size=0.8)

#2.
model = Sequential()
model.add(Dense(8,input_dim=8))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3.
model.compile(loss='mae', optimizer='adam')
start_time = time.time()
model.fit(X_train, y_train, epochs=1000, batch_size=600
          , validation_split=0.2
          , verbose=1)
end_time = time.time()

#4.
loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
results = model.predict(X)

r2 = r2_score(y_test, y_predict)
print("R2 score : ", r2)
print("loss : " , loss)
print("소요 시간 : ", round(end_time - start_time, 2), "seconds")