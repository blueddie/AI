from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten

#1.
datasets = load_wine()

X = datasets.data
y = datasets.target

print(X.shape, y.shape) #(178, 13) (178,)
# print(pd.value_counts(y))

y = y.reshape(-1,1)
print(y.shape)  #(178, 1)

ohe = OneHotEncoder(sparse=True)
y = ohe.fit_transform(y).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13, train_size=0.8)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)

X = X.reshape(-1, 13, 1)

#2. 모델
model = Sequential()
model.add(Conv1D(8, 2,input_shape=(13, 1)))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(3, activation='softmax'))

#3
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=40
                   , verbose=1
                   , restore_best_weights=True
                   )

model.fit(X_train, y_train, epochs=300, batch_size=1
        , validation_split=0.2
        , callbacks=[es]
        )

results = model.evaluate(X_test, y_test)
print("loss : ", results[0])
print("accuracy : " , results[1])

print("-----------------------")

y_predict = model.predict(X_test)

y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

def ACC(aaa, bbb):
    return accuracy_score(y_test, y_predict)

acc = ACC(y_test, y_predict)
print("accuracy : ", acc)
