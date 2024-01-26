import numpy as np
import time
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D, GlobalAveragePooling2D, Input
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import os
from keras.preprocessing.image import ImageDataGenerator

#1
np_path = 'c:/_data/_save_npy/'

x = np.load(np_path + 'keras39_3_cat_dog_x_np.npy')
y = np.load(np_path + 'keras39_3_cat_dog_y_np.npy')
test = np.load(np_path + 'kaggle_cat_dog_submission_np.npy')

# print(x.shape, y.shape) #(19997, 120, 120, 3) (19997,)


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


x_train, x_test, y_train, y_test = train_test_split(x, y ,random_state=1452, train_size=0.75, stratify=y)

# print(x_train.shape, x_test.shape)  #(17197, 120, 120, 3) (2800, 120, 120, 3)

train_datagen = ImageDataGenerator(
    # rescale=1./255
    horizontal_flip=True
    , vertical_flip=True
    , width_shift_range=0.2
    , height_shift_range=0.2
    , rotation_range=30
    , zoom_range=0.2
    , shear_range=20
    , fill_mode='nearest' # default: nearest
)

augument_size = 10

randidx = np.random.randint(x_train.shape[0], size=augument_size)

x_augumented = x_train[randidx].copy() 
y_augumented = y_train[randidx].copy()

x_augumented = train_datagen.flow(
    x_augumented, y_augumented
    , batch_size=augument_size
    , shuffle=False
    , save_to_dir='C:\\_data\\temp\\cat_dog\\'
    ).next()[0]

'''
# print(x_augumented.shape)   #(8000, 120, 120, 3)
# print(y_augumented.shape)   #(8000,)

x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))

# print(y_train.shape)    #(25197,)
# print(y_test.shape)     #(2800,)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
# print(y_train.shape)    #(25197, 1)
# print(y_test.shape)     #(2800, 1)

#3
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])

fit_start_time = time.time()
model.fit(x_train, y_train, batch_size=64, verbose=1, epochs=1000, validation_split=0.2, callbacks=[es])
fit_end_time = time.time()

# 에측
results = model.evaluate(x_test, y_test)

loss = results[0]
acc = results[1]


submit = model.predict(test)
submission = []
submission = submit
threshold = 0.5
binary_submission = (submission > threshold).astype(int)
print(binary_submission.shape) #(5000, 1)
print(binary_submission)

binary_submission = binary_submission.reshape(-1)

#
folder_path = 'C:\\_data\\image\\cat_and_dog\\Test\\test'
file_list = os.listdir(folder_path)
file_names = np.array([os.path.splitext(file_name)[0] for file_name in file_list])

# folder_path = '사진\\폴더\\경로'
# file_list = os.listdir(folder_path)
# file_names = np.array([os.path.splitext(file_name)[0] for file_name in file_list])

y_submit = pd.DataFrame({'id' : file_names, 'Target' : binary_submission})

# print(y_submit['Target'])
csv_path = 'C:\\_data\\kaggle\\cat_dog\\'


date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   

y_submit.to_csv(csv_path + date + "_acc_" + str(round(acc, 4)) + ".csv", index=False)

print('loss : ' , results[0])
print('acc : ' , results[1])
print("걸린 시간 : ", round(fit_end_time - fit_start_time, 3), '초')

file_acc = str(round(results[1], 6))
import datetime
date = datetime.datetime.now().strftime("%m%d_%H%M")

model.save('C:\\_data\\_save\\models\\kaggle\\cat_dog\\'+ date + '_' + file_acc +'_cnn.hdf5')
'''