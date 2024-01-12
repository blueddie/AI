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
path = "C://_data//dacon//wine//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

items = train_csv['type']

encoder.fit(items)
train_csv['type'] = encoder.transform(items)
test_csv['type'] = encoder.transform(test_csv['type'])
# print('인코당 클래스: ', encoder.classes_)  #['red' 'white']


X = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']


print(y.shape)

# print(X.shape)  #(5497, 12)
# print(y.shape)  #(5497, )

# print(pd.value_counts(y))
# 6    2416
# 5    1788
# 7     924
# 4     186
# 8     152
# 3      26
# 9       5

y = pd.get_dummies(y, dtype='int')
print(y.shape)

# #2
model = Sequential()
model.add(Dense(8, activation='relu', input_dim=12))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(7, activation='softmax'))

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=40
                   , verbose=1
                   , restore_best_weights=True
                   )

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13, train_size=0.8)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=1000, batch_size=32
#         , validation_split=0.2
#         , callbacks=[es]
#         )
# results = model.evaluate(X_test, y_test)
# acc = results[1]

def auto() :
    rs = random.randrange(1,9999999)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs, train_size=0.8)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32
        , validation_split=0.2
        , callbacks=[es]
        )

    results = model.evaluate(X_test, y_test)
    acc = results[1]
    y_predict = model.predict(test_csv)
    y_submit = np.argmax(y_predict, axis=1)
    submission_csv['quality'] = y_submit
    return acc

acc = auto()
print(acc)

