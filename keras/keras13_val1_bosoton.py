from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터
datasets = load_boston()

X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=3)

#2 모델
model = Sequential()
model.add(Dense(8, input_dim=13))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#.컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
start_time = time.time()

model.fit(X_train, y_train, epochs=900, batch_size=80, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)

r2 = r2_score(y_test, y_predict)
print("R2 score : ", r2)
print("loss : " , loss)