from sklearn.datasets import fetch_california_housing
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import datetime, time

#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.7)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2 함수형
input1 = Input(shape=(8,))
dense1 = Dense(8)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(16)(drop1)
dense3 = Dense(32)(dense2)
dense4 = Dense(8)(dense3)
drop2 = Dropout(0.3)(dense4)
dense5 = Dense(4)(drop2)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)


date = datetime.datetime.now().strftime("%m%d_%H%M")
path = '..\\_data\_save\\MCP\\california\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'california_', date, '_' ,filename])

#3.
from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=1000
                   , verbose=1
                   , restore_best_weights=True
                   )
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)


model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

st = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64
                 , validation_split=0.2
                 , callbacks=[es, mcp])
et = time.time()

#4.
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
results = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("R2 score : ", r2)
print("loss : " , loss)

print("걸린 시간 : ", et - st)

# cpu : 142.82931280136108
# gpu : 312.76387333869934

