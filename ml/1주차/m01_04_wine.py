from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import LinearSVC

#1.
datasets = load_wine()

X = datasets.data
y = datasets.target

# print(X.shape, y.shape) #(178, 13) (178,)
# print(pd.value_counts(y))

# y = y.reshape(-1,1)
# print(y.shape)  #(178, 1)

# ohe = OneHotEncoder(sparse=True)
# y = ohe.fit_transform(y).toarray()

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=13, train_size=0.8, stratify=y)

#2. 모델
# model = Sequential()
# model.add(Dense(8, input_dim=13))
# model.add(Dense(16))
# model.add(Dense(32))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(3, activation='softmax'))
model = LinearSVC(C=100)        # C가 커질수록 직선에 가깝고 작을수록 굴곡이 진다

#3
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
# results = model.evaluate(x_test, y_test)
# print("loss : ", results[0])
# print("accuracy : " , results[1])
model.fit(x_train, y_train)
print("-----------------------")

# y_predict = model.predict(x_test)

# y_test = np.argmax(y_test, axis=1)
# y_predict = np.argmax(y_predict, axis=1)
#4
results = model.score(x_test, y_test)
print('model.score : ', results)     
y_predict = model.predict(x_test)
# loss = model.loss_function_
def ACC(aaa, bbb):
    return accuracy_score(y_test, y_predict)

acc = ACC(y_test, y_predict)
print("accuracy : ", acc)
# loss :  0.19696640968322754
# accuracy :  0.9444444179534912

# acc :  0.9
# model.score :  0.9166666666666666
# accuracy :  0.9166666666666666
