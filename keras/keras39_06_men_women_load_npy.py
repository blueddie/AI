import numpy as np
import time
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D, GlobalAveragePooling2D, Input
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime


#1
np_path = 'c:/_data/_save_npy/'

x = np.load(np_path + 'keras39_5_men_women_x_np.npy')
y = np.load(np_path + 'keras39_5_men_women_y_np.npy')
test = np.load(np_path + 'keras39_5_men_women_test_np.npy')

# print(x.shape)      #(3807, 120, 120, 3)
# print(y.shape)      #(3807,)    
# print(test.shape)   #(1, 120, 120, 3)

#2
model = Sequential()
model.add(Conv2D(16, (2,2) , strides=2, input_shape=(120, 120, 3)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.4))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(units=64))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3 
x_train, x_test, y_train, y_test = train_test_split(x, y ,random_state=1124, train_size=0.88, stratify=y)

es = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=0, restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])

fit_start_time = time.time()

model.fit(x_train, y_train, batch_size=32, verbose=1, epochs=1000, validation_split=0.2, callbacks=[es])

fit_end_time = time.time()

results = model.evaluate(x_test, y_test)

loss = results[0]
acc = results[1]

print('loss : ' , results[0])
print('acc : ' , results[1])
print("걸린 시간 : ", round(fit_end_time - fit_start_time, 3), '초')

y_predict = model.predict(test)
print(np.around(y_predict))

date = datetime.datetime.now().strftime("%m%d_%H%M")
file_acc = str(round(results[1], 6))
model.save('C:\\_data\\_save\\models\\kaggle\\men_women\\'+ date + '_' + file_acc +'_cnn.hdf5')


# loss :  0.5378432273864746
# acc :  0.7308533787727356
# 걸린 시간 :  16.562 초
# [[0.]]