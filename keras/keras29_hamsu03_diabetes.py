from sklearn.datasets import load_diabetes
from keras.models import Sequential, save_model, Model
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import datetime

#1. 데이터
datasets = load_diabetes()
X = datasets.data
y = datasets.target


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1226, train_size=0.85)

scaler = StandardScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)

date = datetime.datetime.now().strftime("%m%d_%H%M")
path = '..\\_data\_save\\MCP\\california\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'california_', date, '_' ,filename])



#2  순차적
# model = Sequential()
# model.add(Dense(8, input_dim=10))
# model.add(Dropout(0.2))
# model.add(Dense(16))
# model.add(Dense(32))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(1))

#2  함수형

input1 = Input(shape=(10,))
dense1 = Dense(8)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(16)(drop1)
dense3 = Dense(32)(dense2)
dense4 = Dense(8)(dense3)
dense5 = Dense(4)(dense4)
output1 = Dense(4)(dense5)
model = Model(inputs=input1, outputs=output1)



from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=100
                   , verbose=1
                   , restore_best_weights=True
                   )

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

# #3
model.compile(loss='mae', optimizer='adam')

model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2, callbacks=[es, mcp])
# #4

loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)

r2 = r2_score(y_test, y_predict)
results = model.predict(X)

# print("R2 score : ", r2)
print("loss : " , loss)
