import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential, load_model
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

#3.

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)

model = load_model('..\\_data\\_save\\MCP\\keras26_cancer.hdf5')

#4.
y_predict = np.rint(model.predict(X_test))
loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
results = model.predict(X)

print("loss : " , loss)

# loss :  [0.1443520188331604, 0.9532163739204407]