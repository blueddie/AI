from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

#1
datasets = fetch_covtype()

X = datasets.data
y = datasets.target

# print(y) #5
print(X.shape, y.shape)
print(pd.value_counts(y))


print(X.shape, y.shape) #(581012, 54) (581012, 1)

#keras
# from keras.utils import to_categorical
# y_ohe = to_categorical(y)
# y = y_ohe
# print(y)
# print("값은 ", y[0][5])


# print(y.shape)

# pandas
y = pd.get_dummies(y)
# print(y)

# sklearn
# y = y.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=True)
# y = ohe.fit_transform(y).toarray()

X = X.reshape(-1, 54, 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=15152, train_size=0.8)

#2
model = Sequential()
model.add(Conv1D(16, 2, input_shape=(54,1)))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(28))
model.add(Dense(7, activation='softmax'))

#3
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy'
                   , mode='max'
                   , patience=50
                   , verbose=1
                   , restore_best_weights=True
                   )
model.fit(X_train, y_train, epochs=5000, batch_size=100
        , validation_split=0.2
        , callbacks=[es]
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