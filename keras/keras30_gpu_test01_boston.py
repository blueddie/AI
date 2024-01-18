#dropout

from sklearn.datasets import load_boston
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import datetime, time

datasets = load_boston()

#1. 데이터
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.7)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2 모델 구성    #( 함수형 )
input1 = Input(shape=(13,))
dense1 = Dense(10)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(10)(drop1)
dense3 = Dense(10)(dense2)
drop2 = Dropout(0.3)(dense3)
dense4 = Dense(10)(drop2)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)

# date = datetime.datetime.now()  #2024-01-17 10:53:47.961440
date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   
print(date)
# print(date1)

path = '..\\_data\_save\\MCP\\boston\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #   04d : 4자리 정수로 표현 int   .4f : 소수점 아래 4자리 표현 float
filepath = ''.join([path, 'boston_', date, '_' ,filename])
# filename = filepath + str(date1) + ".hdf5"
print(filepath)

# # #3.컴파일, 훈련
es = EarlyStopping(monitor='val_loss', mode='min', patience=1000, verbose=1, restore_best_weights=True)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=False, filepath=filepath)


model.compile(loss='mae', optimizer='adam')

st = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es, mcp])
et = time.time()


# #4. 평가, 예측

print('=========================   1. 기본 출력   ====================================')
loss = model.evaluate(x_test, y_test, verbose=0)
y_predict = model.predict(x_test, verbose=0)
r2 = r2_score(y_test, y_predict)

print("R2 score : ", r2)
print("loss : " , loss)
print('걸린 시간 :' , et - st)


# cpu : 32.967416286468506초 
# gpu : 41.81936287879944