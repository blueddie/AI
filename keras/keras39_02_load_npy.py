import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D, GlobalAveragePooling2D, Input
# import os
from sklearn.model_selection import train_test_split

#1 데이터
# train_datagen = ImageDataGenerator(
#     rescale=1./255
# )

# test_datagen = ImageDataGenerator(
#     rescale=1./255,
# )


# path_train = 'c:\\_data\\image\\brain\\train\\'
# path_test = 'c:\\_data\\image\\brain\\test\\'

# Xy_train = train_datagen.flow_from_directory(
#     path_train
#     , target_size=(100, 100)
#     , batch_size= 10
#     , class_mode='binary'
#     , color_mode='grayscale' # default
#     , shuffle='True'
#     #Found 20000 images belonging to 2 classes.
# )

# Xy_test = test_datagen.flow_from_directory(
#     path_test
#     , target_size=(100,100)
#     , batch_size=10
#     , class_mode='binary'
#     , color_mode='grayscale'
#     # , shuffle='True'
#     # Found 120 images belonging to 2 classes.
# )

# st = time.time()
# 전체 데이터셋을 하나의 데이터로 만들기

# X = []
# y = []

# for i in range(len(Xy_train)):
#     images, labels = Xy_train.next()
#     X.append(images)
#     y.append(labels)

# # all_images와 all_labels을 numpy 배열로 변환하면 하나의 데이터로 만들어진 것입니다.
# X = np.concatenate(X, axis=0)
# y = np.concatenate(y, axis=0)

# et = time.time()

# print('이미지 변환에 걸린 시간 : ', round(et - st, 3), ' 초')

# print(X.shape)    # (160, 100, 100, 1)
# print(y.shape)    # (160,)

# X_test = Xy_test[0][0]
# y_test = Xy_test[0][1]

# #1 
np_path = 'c:/_data/_save_npy/'
# np.save(np_path + 'keras39_1_x_train.npy', arr=Xy_train[0][0])
# np.save(np_path + 'keras39_1_y_train.npy', arr=Xy_train[0][1])
# np.save(np_path + 'keras39_1_x_test.npy', arr=Xy_test[0][0])
# np.save(np_path + 'keras39_1_y_test.npy', arr=Xy_test[0][1])

X_train = np.load(np_path + 'keras39_1_x_train.npy')
X_test = np.load(np_path + 'keras39_1_x_test.npy')

y_train = np.load(np_path + 'keras39_1_y_train.npy')
y_test = np.load(np_path + 'keras39_1_y_test.npy')
# print(X_train)
# print(X_train.shape)

#2
model = Sequential()
model.add(Conv2D(9, (2,2) , strides=2, input_shape=(100, 100, 1)))
model.add(Conv2D(64, (3,3)))
model.add(Dropout(0.4))
model.add(MaxPooling2D())
model.add(Conv2D(15, (3,3)))
model.add(Flatten())
model.add(Dense(units=8))
model.add(Dense(7, input_shape=(8,)))
model.add(Dense(6))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3
# X_train, X_test, y_train, y_test = train_test_split(X, y ,random_state=1452, train_size=0.86, stratify=y)
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=0, restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])

fit_st = time.time()

model.fit(X_train, y_train
                    , batch_size=32 # fit_generator 에서는 에러, fit에서는 안 먹힘
                    , verbose=1
                    , epochs=1000
                    # , steps_per_epoch=16 # 전체 데이터 / 160 / batch_size == 16 넘어가면 데이터 손실
                    , validation_split=0.2
                    # , validation_data=Xy_test
                    , callbacks=[es])
#   UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version.
#   Please use `Model.fit`, which supports generators.

fit_et = time.time()

#4
results = model.evaluate(X_test, y_test)
print(results)
print('loss : ' , results[0])
print('acc : ' , results[1])
print("걸린 시간 : ", round(fit_et - fit_st, 3), '초')


file_acc = str(round(results[1], 4))

# import datetime
# date = datetime.datetime.now().strftime("%m%d_%H%M")
# model.save('C:\\_data\\_save\\MCP\\cat_dog\\'+ date + '_' + file_acc +'_cnn.hdf5')
