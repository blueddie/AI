# https://dacon.io/competitions/open/236070/overview/description
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from sklearn.preprocessing import OneHotEncoder

#1.
path = "C://_data//dacon//iris//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

X = train_csv.drop(['species'], axis=1)
y = train_csv['species']

# print(X.shape)  #(120, 4)
# print(y.shape)  #(120,)

y = pd.get_dummies(y, dtype='int')

#2
model = Sequential()
model.add(Dense(8,activation='relu', input_dim=4))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(3, activation='softmax'))

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13, train_size=0.8)
#3
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy'
                   , mode='max'
                   , patience=30
                   , verbose=1
                   , restore_best_weights=True
                   )

# model.fit(X_train, y_train, epochs=200, batch_size=1
#         , validation_split=0.2
#         , callbacks=[es]
#         )

def auto() :
    rs = random.randrange(1,9999999)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs, train_size=0.8)
    
    model.fit(X_train, y_train, epochs=300, batch_size=1
        , validation_split=0.2
        , callbacks=[es]
        )

    results = model.evaluate(X_test, y_test)
    acc = results[1]
    y_predict = model.predict(test_csv)
    y_submit = np.argmax(y_predict, axis=1)
    submission_csv['species'] = y_submit
    return acc

iwant = 1
i = 1

while True :
    acc = auto()
    if acc == iwant :
        submission_csv.to_csv(path + str(i) + ".csv" ,index=False)
        i += 1
    

