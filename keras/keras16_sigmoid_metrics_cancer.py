import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import time


#1. 데이터
datasets = load_breast_cancer()
# print(datasets)

# print(datasets.DESCR)
# print(datasets.feature_names)

X = datasets.data
y = datasets.target

# print(X.shape)  #(569, 30)
# print(y.shape)  #(569,)

# print(np.unique(y))

# 0과 1의 갯수가 몇개인지 찾는 법을 찾아라!!
#numpy
print(np.unique(y, return_counts=True))

#pandas
# print(pd.DataFrame(y).value_counts()) #모두 같다
# print(pd.Series(y).value_counts())    #모두 같다
print(pd.value_counts(y))   #모두 같다



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=53, train_size=0.8)

def ACC(aaa, bbb):
    return accuracy_score(aaa,bbb)



#2.
model = Sequential()
model.add(Dense(8,input_dim=30))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1, activation='sigmoid')) # bindary_crossentropy에서는 마지막 layer activation='sigmoid'를 써줘야 한다


from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=50
                   , verbose=1
                   , restore_best_weights=True
                   )




# y_predict = y_predict.round()

# acc = ACC(y_test, y_predict)

# print(acc)

# print(y_test)
# print(y_predict)



#3.
model.compile(loss='binary_crossentropy', optimizer='adam'
              , metrics=['accuracy']
              )
start_time = time.time()
model.fit(X_train, y_train, epochs=10, batch_size=64
          , validation_split=0.2
          , verbose=1
          , callbacks=[es]
          )
end_time = time.time()

#4.
y_predict = np.rint(model.predict(X_test))
loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
results = model.predict(X)


print(y_predict)

print("loss : " , loss)
print("소요 시간 : ", round(end_time - start_time, 2), "seconds")

