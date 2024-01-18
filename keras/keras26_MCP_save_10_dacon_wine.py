# https://dacon.io/competitions/open/235610/overview/description
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


encoder = LabelEncoder()
#1.
csv_path = "C://_data//dacon//wine//"

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

items = train_csv['type']

encoder.fit(items)
train_csv['type'] = encoder.transform(items)
test_csv['type'] = encoder.transform(test_csv['type'])
# print('인코당 클래스: ', encoder.classes_)  #['red' 'white']

X = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']

y = pd.get_dummies(y, dtype='int')
print(y.shape)

# #2
model = Sequential()
model.add(Dense(8, input_dim=12))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(7, activation='softmax'))

from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_accuracy'
                   , mode='max'
                   , patience=120
                   , verbose=1
                   , restore_best_weights=True
                   )


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13, train_size=0.8, stratify=y)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)
test_csv = scaler.transform(test_csv)    

import datetime

date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   
path = '..\\_data\_save\\MCP\\dacon_wine\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'dacon_wine_', date, '_' ,filename])


mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1500, batch_size=64
    , validation_split=0.2
    , callbacks=[es, mcp])



results = model.evaluate(X_test, y_test)
acc = results[1]
loss = results[0]
y_predict = model.predict(test_csv)
y_submit = np.argmax(y_predict, axis=1)
submission_csv['quality'] = y_submit + 3


submission_csv.to_csv(csv_path + date + str(round(acc,2)) + ".csv", index=False)