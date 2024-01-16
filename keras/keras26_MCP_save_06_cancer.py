import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score


#1. 데이터
datasets = load_breast_cancer()


X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=53, train_size=0.7)

def ACC(aaa, bbb):
    return accuracy_score(aaa,bbb)

#2.
model = Sequential()
model.add(Dense(8,input_dim=30))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1, activation='sigmoid'))

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=60
                   , verbose=1
                   , restore_best_weights=True
                   )

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath='..\\_data\\_save\\MCP\\keras26_cancer.hdf5')

#3.
model.compile(loss='binary_crossentropy', optimizer='adam'
              , metrics=['accuracy']
              )
model.fit(X_train, y_train, epochs=1000, batch_size=64
          , validation_split=0.2
          , verbose=1
          , callbacks=[es, mcp]
          )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)


#4.
y_predict = np.rint(model.predict(X_test))
loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
results = model.predict(X)

print("loss : " , loss)

# loss :  [0.1443520188331604, 0.9532163739204407]