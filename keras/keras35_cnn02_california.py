from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import datetime

#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.7)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2.
model = Sequential()
model.add(Dense(8,input_dim=8))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dropout(0.3))
model.add(Dense(4))
model.add(Dense(1))

date = datetime.datetime.now().strftime("%m%d_%H%M")
path = '..\\_data\_save\\MCP\\california\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'california', date, '_' ,filename])

#3.
from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=20
                   , verbose=1
                   , restore_best_weights=True
                   )
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)


model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])


hist = model.fit(x_train, y_train, epochs=120, batch_size=64
                 , validation_split=0.2
                 , callbacks=[es, mcp])

#4.
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
results = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("R2 score : ", r2)
print("loss : " , loss)