from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
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

print(x.shape)  #(20640, 8)
x_train = x_train.reshape(-1, 4, 2)
x_test = x_test.reshape(-1, 4, 2)
#2.
model = Sequential()
model.add(Conv1D(8, kernel_size=2 ,activation='relu',input_shape=(4, 2)))
model.add(Flatten())
# model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

date = datetime.datetime.now().strftime("%m%d_%H%M")
path = '..\\_data\_save\\MCP\\california\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'california', date, '_' ,filename])

#3.
from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=40
                   , verbose=1
                   , restore_best_weights=True
                   )
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)


model.compile(loss='mae', optimizer='adam')


hist = model.fit(x_train, y_train, epochs=120, batch_size=64
                 , validation_split=0.2
                 , callbacks=[es])

#4.
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("R2 score : ", r2)
print("loss : " , loss)


#Conv1D
# R2 score :  0.7297039610793643
# loss :  0.3825446367263794