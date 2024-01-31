import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D, GlobalAveragePooling2D, Input, LSTM, Bidirectional, SimpleRNN
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
# import os
from sklearn.model_selection import train_test_split


np_path = 'c:\\_data\\_save_npy\\'

x = np.load(np_path + 'kaggle_jena_x720.npy')
y = np.load(np_path + 'kaggle_jena_y720.npy')

x = x.astype(np.float32)
y = y.astype(np.float32)


print(x.shape)  #(420545, 6, 14)
print(y.shape)  #(420545,)


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, train_size=0.84)


x_train_flattened = x_train.reshape(x_train.shape[0], -1)
# print(X_train_flattened.shape)
x_test_flattened = x_test.reshape(x_test.shape[0], -1)


scaler = MinMaxScaler()
scaler.fit(x_train_flattened)
scaled_train = scaler.transform(x_train_flattened)
scaled_test = scaler.transform(x_test_flattened)

x_train = scaled_train.reshape(x_train.shape)
x_test = scaled_test.reshape(x_test.shape)




#2
# model = Sequential()
# model.add(Bidirectional(LSTM(units=64), input_shape=(6, 14))) #timesteps, features
# model.add(Dense(128,activation='relu'))
# model.add(Dense(77,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(25,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(1))


model = Sequential()
model.add(SimpleRNN(units=2, input_shape=(720, 14))) #timesteps, features
model.add(Dense(8,activation='relu'))
model.add(Dense(14,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))
model.summary()

#3
# with tf.device('/CPU:0'):

es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1, restore_best_weights=True)
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=7000, batch_size=2048, callbacks=[es], validation_split=0.2)

#4
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('loss : ', results)
print('r2 : ', r2 )

# loss :  0.1111978068947792
# r2 :  0.9980912929211749


# loss :  0.05431775003671646
# r2 :  0.9990676368807793

# loss :  0.059629276394844055
# r2 :  0.9989764646165151