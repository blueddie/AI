import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

def ACC(aaa, bbb):
    return accuracy_score(aaa,bbb)

#1 . 데이터
datasets = load_iris()
# print(datasets)
# print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=13, train_size=0.8, stratify=y)

#2
# model = Sequential()
# model.add(Dense(8, input_dim=4))
# model.add(Dense(16))
# model.add(Dense(32))
# model.add(Dense(4))
# model.add(Dense(3, activation='softmax'))
model = LinearSVC(C=100)        # C가 크면 training포인트를 정확히 구분(굴곡지다), C가 작으면 직선에 가깝다.

#3 컴파일, 훈련
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=150, batch_size=1, validation_split=0.2)
model.fit(x_train, y_train)

#4
# results = model.evaluate(x_test, y_test)
results = model.score(x_test, y_test)
print('model.score : ', results)            # 분류에서는 accuracy, 회귀에서는 r2



y_predict = model.predict(x_test)
print(y_predict)
# [2 1 2 2 0 2 0 0 1 0 1 1 2 0 0 1 1 2 1 0 1 0 2 1 2 0 2 0 2 1]
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict, y_test)
print('acc : ', acc)
# print(y_test)

# # y_test = np.argmax(y_test, axis=1)
# # y_predict = np.argmax(y_predict, axis=1)
# print(y_test.shape)
# print(y_predict.shape)

# acc = ACC(y_test, y_predict)
# print("accuracy_score : " , acc)

# print("loss : ", loss[0])
# print("accuracy : ", loss[1])

# results = model.predict([[5.9, 3,  5.1, 1.8]])
# print(results)
# max = results[0][0]
# max_index = 0
# for i in range(2):
#     if max < results[0][i + 1] :
#         max = results[0][i + 1]
#         max_index = i + 1 
# results = max_index
# print(results)
# (1, 3)
# results.reshape()
# print(results.shape)    #(1, 3)

# # # [5.1, 3.5, 1.4, 0.2]
# # # [4.9, 3,  1.4, 0.2] 100
# # # [5.9 3.  5.1 1.8] 001

# # # loss :  0.3693692088127136
# # # accuracy :  0.9333333373069763
# # # 1/1 [==============================] - 0s 17ms/step
# # [[0.02058523 0.26513213 0.71428263]]