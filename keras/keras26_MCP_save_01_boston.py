from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

datasets = load_boston()

#1. 데이터
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.7)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2 모델 구성
model = Sequential()
model.add(Dense(8, input_dim=13))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   
path = '..\\_data\_save\\MCP\\boston\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'boston_', date, '_' ,filename])



#.컴파일, 훈련

es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=100
                   , verbose=1
                   , restore_best_weights=True
                   )

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=80, validation_split=0.2, callbacks=[es, mcp])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
results = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("R2 score : ", r2)
print("loss : " , loss)

# R2 score :  0.7087847879357653
# R2 score :  0.7087847879357653
# loss :  3.243027925491333
# loss :  3.243027925491333
