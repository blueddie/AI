from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

#1
datasets = fetch_covtype()

X = datasets.data
y = datasets.target

# print(y) #5
# print(X.shape, y.shape)
# print(pd.value_counts(y))


# print(X.shape, y.shape) #(581012, 54) (581012, 1)


# y = pd.get_dummies(y)
# print(y)

# sklearn
# y = y.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=True)
# y = ohe.fit_transform(y).toarray()

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=15152, train_size=0.8)

#2
# model = Sequential()
# model.add(Dense(54, input_dim=54))
# model.add(Dense(120))
# model.add(Dense(28))
# model.add(Dense(7, activation='softmax'))
model = LinearSVC(C=100)

#3
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train)

# from keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_accuracy'
#                    , mode='max'
#                    , patience=50
#                    , verbose=1
#                    , restore_best_weights=True
#                    )
# model.fit(x_train, y_train, epochs=5000, batch_size=100
#         , validation_split=0.2
#         , callbacks=[es]
#         )

#4
# results = model.evaluate(x_test, y_test)
# print('loss : ', results[0])
# print('accuracy : ', results[1])
results = model.score(x_test, y_test)
print('model.score : ', results)   
y_predict = model.predict(x_test)
print(y_predict)
# y_predict = model.predict(x_test)

# print(y_test)
# print(y_predict)

# y_test = np.argmax(y_test, axis=1)
# y_predict = np.argmax(y_predict, axis=1)
# print(y_test)
# print(y_predict)


def ACC(aaa, bbb):
    return accuracy_score(aaa,bbb)
acc = accuracy_score(y_predict, y_test)
print('acc : ', acc)

# acc = ACC(y_test, y_predict)
# print("accuracy_score : ", acc )

# accuracy_score :  0.7186819617393699
# accuracy_score :  0.7237678889529531
