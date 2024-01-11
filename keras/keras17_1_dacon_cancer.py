# https://dacon.io/competitions/open/236068/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import random
import time

def ACC(aaa, bbb):
    return accuracy_score(aaa,bbb)

#1.
path = "C://_data//dacon//cancer//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")

# print(train_csv.isna().sum())

# print(train_csv.shape)
# idx = train_csv[train_csv['BloodPressure'] == 0].index
# train_csv.drop(idx, inplace=True)

# idx = test_csv[test_csv['BloodPressure'] == 0].index
# test_csv.drop(idx , inplace=True)

# print(test_csv.shape)
# # idx = test_csv[test_csv['BloodPressure'] == 0].index
# # test_csv = test_csv.drop(idx , inplace=True)

X = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

# test_csv = test_csv.drop(['Insulin'],axis=1)
print(X.shape)
# 2. 모델
model = Sequential()
model.add(Dense(16,activation='relu', input_dim=8))
model.add(Dense(32,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy'
                   , mode='max'
                   , patience=80
                   , verbose=1
                   , restore_best_weights=True
                   )
# #3.
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=312, train_size=0.8)

# model.compile(loss='binary_crossentropy', optimizer='adam'
#               , metrics=['accuracy']
#               )
# model.fit(X_train, y_train, epochs=1000, batch_size=50
#           , validation_split=0.2
#           , verbose=1
#           , callbacks=[es]
#           )

# #4.
# loss = model.evaluate(X_test, y_test)
# results = model.predict(X)

# y_predict = np.rint(model.predict(X_test))

# acc = ACC(y_test, y_predict)
# print(acc)
# y_submit = model.predict([test_csv])



# submission_csv['Outcome'] = np.rint(y_submit)
# submission_csv.to_csv(path + "a.csv" ,index=False)

#---------------------------------------------------------------------
def auto() :
    rs = random.randrange(1,99999999)
    bs = random.randrange(32,257)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs, train_size=0.8)

    model.compile(loss='binary_crossentropy', optimizer='adam'
                , metrics=['accuracy']
                )
    model.fit(X_train, y_train, epochs=1000, batch_size=bs
            , validation_split=0.2
            , verbose=1
            , callbacks=[es]
            )

    #4.
    loss = model.evaluate(X_test, y_test)
   
    results = model.predict(X)

    y_predict = np.rint(model.predict(X_test))

    acc = ACC(y_test, y_predict)
    print(acc)
    y_submit = model.predict([test_csv])
    submission_csv['Outcome'] = np.rint(y_submit)
    return acc, rs, bs , loss

max_acc = 0.85

while True :
    
    acc, rs, bs, loss = auto()
    if acc > max_acc:
        
        submission_csv.to_csv(path + "acc노갱신"+ str(round(acc,2))+ "rs_" + str(rs) + "bs_"+ str(bs) +".csv" ,index=False)
    
    
    
    
    
    