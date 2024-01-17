from sklearn.datasets import load_boston
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

datasets = load_boston()

#1. 데이터
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.7)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

# # #3.컴파일, 훈련
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath='..\\_data\\_save\\MCP\\keras25_MCP3.hdf5')

model.compile(loss='mae', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es, mcp])

model.save('..\\_data\\_save\\keras25_3_save_model_.hdf5')

model = load_model('..\\_data\_save\\MCP\\keras25_MCP1.hdf5')

# #4. 평가, 예측

print('=========================   1. 기본 출력   ====================================')
loss = model.evaluate(x_test, y_test, verbose=0)
y_predict = model.predict(x_test, verbose=0)
r2 = r2_score(y_test, y_predict)

print("R2 score : ", r2)
print("loss : " , loss)

print('=========================  2. load_model 출력    ==================================')

model2 = load_model('..\\_data\\_save\\keras25_3_save_model_.hdf5')

loss2 = model2.evaluate(x_test, y_test, verbose=0)
y_predict2 = model2.predict(x_test, verbose=0)
r2 = r2_score(y_test, y_predict2)

print("R2 score : ", r2)
print("loss : " , loss)


print('=========================  3. MCP 모델 출력    ==================================')

model3 = load_model('..\\_data\\_save\\keras25_3_save_model_.hdf5')

loss3 = model3.evaluate(x_test, y_test, verbose=0)
y_predict3 = model3.predict(x_test, verbose=0)
r2 = r2_score(y_test, y_predict3)

print("R2 score : ", r2)
print("loss : " , loss)