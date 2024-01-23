import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import time
from sklearn.preprocessing import OneHotEncoder


#1. 데이터
datasets = load_breast_cancer()

X = datasets.data
y = datasets.target

print(X.shape)  #(569, 30)
print(y.shape)  #(569,)

X = X.reshape(X.shape[0], 10, 3, 1)
print(X.shape)
y = y.reshape(-1, 1)

print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=53, train_size=0.8)

def ACC(aaa, bbb):
    return accuracy_score(aaa,bbb)

#2.
model = Sequential()
model.add(Conv2D(97, (2,1), activation='relu', input_shape=(10, 3, 1)))
model.add(Conv2D(32, (2, 1), activation='relu', padding='same'))
model.add(Conv2D(160, (2,1), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(Conv2D(120, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # bindary_crossentropy에서는 마지막 layer activation='sigmoid'를 써줘야 한다


from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=60
                   , verbose=1
                   , restore_best_weights=True
                   )

#3.
model.compile(loss='binary_crossentropy', optimizer='adam'
              , metrics=['accuracy']
              )
start_time = time.time()
model.fit(X_train, y_train, epochs=10, batch_size=32
          , validation_split=0.2
          , verbose=1
          , callbacks=[es]
          )


st = time.time()

model.fit(X_train, y_train, batch_size=64, verbose=1, epochs=110, validation_split=0.2, callbacks=[es])
end_time = time.time()

#4.
results = model.evaluate(X_test, y_test)
acc = results[1]
loss = results[0]

print("loss : ", loss)
print("acc : ", acc)
print("소요 시간 : ", round(end_time - start_time, 2), "seconds")

# loss :  0.0665399432182312
# acc :  0.9824561476707458
# 소요 시간 :  5.64 seconds
