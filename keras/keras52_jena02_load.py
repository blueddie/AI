import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D, GlobalAveragePooling2D, Input, LSTM, Bidirectional
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# import os
from sklearn.model_selection import train_test_split


np_path = 'c:\\_data\\_save_npy\\'

x = np.load(np_path + 'kaggle_jena_x.npy')
y = np.load(np_path + 'kaggle_jena_y.npy')

x = x.astype(np.float32)
y = y.astype(np.float32)


print(x.shape)  #(420545, 6, 14)
print(y.shape)  #(420545,)


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, train_size=0.84)

#2
model = Sequential()
model.add(Bidirectional(LSTM(units=64), input_shape=(5, 14))) #timesteps, features
model.add(Dense(128,activation='relu'))
model.add(Dense(77,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

model.summary()

#3
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1, restore_best_weights=True)
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=7000, batch_size=1024, callbacks=[es], validation_split=0.2)

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
