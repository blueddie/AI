import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import time
from sklearn.svm import LinearSVC



#1. 데이터
datasets = load_breast_cancer()

X = datasets.data
y = datasets.target

#numpy
# print(np.unique(y, return_counts=True))

#pandas
# print(pd.DataFrame(y).value_counts()) #모두 같다
# print(pd.Series(y).value_counts())    #모두 같다
# print(pd.value_counts(y))   #모두 같다

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=53, train_size=0.8, stratify=y)

def ACC(aaa, bbb):
    return accuracy_score(aaa,bbb)

#2.
# model = Sequential()
# model.add(Dense(8,input_dim=30))
# model.add(Dense(16))
# model.add(Dense(32))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(1, activation='sigmoid')) # bindary_crossentropy에서는 마지막 layer activation='sigmoid'를 써줘야 한다
model = LinearSVC(C=100)


#3.
# model.compile(loss='binary_crossentropy', optimizer='adam'
#               , metrics=['accuracy']
#               )
# start_time = time.time()
# model.fit(X_train, y_train, epochs=10, batch_size=64
#           , validation_split=0.2
#           , verbose=1
#           , callbacks=[es]
#           )
model.fit(x_train, y_train)
#4.
# y_predict = np.rint(model.predict(X_test))
# loss = model.evaluate(X_test, y_test)
# y_predict = model.predict(X_test)
# results = model.predict(X)

results = model.score(x_test, y_test)
print('model.score : ', results)            # 분류에서는 accuracy, 회귀에서는 r2



y_predict = model.predict(x_test)
print(y_predict)
# [2 1 2 2 0 2 0 0 1 0 1 1 2 0 0 1 1 2 1 0 1 0 2 1 2 0 2 0 2 1]
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict, y_test)
print('acc : ', acc)

# acc :  0.9298245614035088