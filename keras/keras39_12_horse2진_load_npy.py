import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D, GlobalAveragePooling2D, Input
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import time


#1 데이터

np_path = 'c:\\_data\\_save_npy\\'

x = np.load(np_path + 'keras39_0125_162907_horse_2진_x_np.npy')
y = np.load(np_path + 'keras39_0125_162907_horse_2진_y_np.npy')

#2
model = Sequential()
model.add(Conv2D(120, (3,3), input_shape=(300, 300, 3)))
model.add(Conv2D(525, (3,3), activation='relu'))
# model.add(MaxPooling2D())
model.add(Conv2D(123, (3,3), activation='relu'))
model.add(Dropout(0.5))
# model.add(MaxPooling2D())
model.add(Dropout(0.5))
# model.add(Conv2D(555, (3,3), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(256, (2,2), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(125, (2,2), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
# model.add(Dense(124))
# model.add(Dense(124))
# model.add(Dense(124))
# model.add(Dense(124))
model.add(Dense(124))
model.add(Dense(124))
model.add(Dense(124))
model.add(Dense(124))
model.add(Dropout(0.5))
model.add(Dense(14))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

'''
#3
x_train, x_test, y_train, y_test = train_test_split(x, y ,random_state=13, train_size=0.75, stratify=y)

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=0, restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])

fit_start_time = time.time()

model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=1000, validation_split=0.2, callbacks=[es])

fit_end_time = time.time()


#4
results = model.evaluate(x_test, y_test)

loss = results[0]
acc = results[1]

print('loss : ' , results[0])
print('acc : ' , results[1])
print("걸린 시간 : ", round(fit_end_time - fit_start_time, 3), '초')

date = datetime.datetime.now().strftime("%m%d_%H%M")
file_acc = str(round(results[1], 6))
model.save('C:\\_data\\_save\\models\\kaggle\\horse_human\\'+ date + '_' + file_acc +'_cnn.hdf5')


# loss :  0.014050924219191074
# acc :  0.9930555820465088
# 걸린 시간 :  58.205 초

# loss :  0.01377693098038435
# acc :  1.0
# 걸린 시간 :  77.531 초





# 서머리 4200만개

'''