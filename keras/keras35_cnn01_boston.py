from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
import datetime

datasets = load_boston()

X = datasets.data
y = datasets.target
print(X.shape)  #(506, 13)
X = X.reshape(-1, 1, 13, 1)
print(X.shape)  #(506, 1, 13, 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, train_size=0.8)
print(X_train.shape)    #(404, 1, 13, 1)
print(X_test.shape)     #(102, 1, 13, 1)
print(y.shape)          #(506,)

#2
model = Sequential()
model.add(Conv2D(97, (1,3), activation='relu', padding='same', input_shape=(1, 13, 1)))
model.add(Conv2D(32, (1, 3), activation='relu', padding='same'))
model.add(Conv2D(160, (1,2), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(Conv2D(120, (1,2), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(23, activation='relu'))
model.add(Dense(1)) 


#3
es = EarlyStopping(monitor='val_loss'
                   , mode='min'
                   , patience=40
                   , verbose=1
                   , restore_best_weights=True
                   )

model.compile(loss='mae', optimizer='adam')
model.fit(X_train, y_train, epochs=500, batch_size=64, validation_split=0.2, callbacks=[es])

loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
results = model.predict(X)

r2 = r2_score(y_test, y_predict)

print("R2 score : ", r2)
print("loss : " , loss)

# R2 score :  0.7615445774567593
# loss :  3.145082473754883

# R2 score :  0.6817796619773973
# loss :  3.125746965408325

# R2 score :  0.6736491113441989
# loss :  3.5278685092926025

# R2 score :  0.7359853474011155
# loss :  2.8940417766571045



