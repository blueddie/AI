from sklearn.datasets import load_diabetes
from keras.models import Sequential, save_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = load_diabetes()
X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1226, train_size=0.85)

scaler = StandardScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)

#2
model = Sequential()
model.add(Dense(8, input_dim=10))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=100
                   , verbose=1
                   , restore_best_weights=True
                   )

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath='..\\_data\\_save\\MCP\\keras26_diabetes.hdf5')

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

# loss :  71.52172088623047