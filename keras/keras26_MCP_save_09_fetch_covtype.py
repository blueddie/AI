from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

#1
datasets = fetch_covtype()

X = datasets.data
y = datasets.target

print(X.shape, y.shape)
print(pd.value_counts(y))


print(X.shape, y.shape) #(581012, 54) (581012, 1)

y = pd.get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=15152, train_size=0.8)

#2
model = Sequential()
model.add(Dense(54, input_dim=54))
model.add(Dense(120))
model.add(Dense(28))
model.add(Dense(7, activation='softmax'))


import datetime
date = datetime.datetime.now().strftime("%m%d_%H%M")
path = '..\\_data\_save\\MCP\\fetch_covtype\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'fetch_covtype', date, '_' ,filename])



#3

from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_accuracy'
                   , mode='max'
                   , patience=50
                   , verbose=1
                   , restore_best_weights=True
                   )
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5000, batch_size=100
        , validation_split=0.2
        , callbacks=[es, mcp]
        )


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)

results = model.evaluate(X_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])


y_predict = model.predict(X_test)

print(y_test)
print(y_predict)

y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
print(y_test)
print(y_predict)


def ACC(aaa, bbb):
    return accuracy_score(aaa,bbb)


acc = ACC(y_test, y_predict)
print("accuracy_score : ", acc )