import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D, GlobalAveragePooling2D, Input
import os
from sklearn.model_selection import train_test_split

train_datagen = ImageDataGenerator(
    rescale=1./255
    # , horizontal_flip=True      # 수평 뒤집기
    # , vertical_flip=True        # 수직 뒤집기
    # , width_shift_range=0.2     # 0-10% 평행 이동하겠다 이동한만큼 0이 채워짐
    # , height_shift_range=0.3    # 0-10% 수직  
    # , rotation_range=7          # 정해진 각도만큼 이미지를 회전
    # , zoom_range=1.3            # 축소 또는 확대
    # , shear_range=0.5           # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    # , fill_mode='nearest'       # 빈자리를 가장 비슷한 색으로 채움
    # , 
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)


path_train = 'c:\\_data\\image\\cat_and_dog\\Train\\'
path_test = 'c:\\_data\\image\\cat_and_dog\\Test\\'

Xy_train = train_datagen.flow_from_directory(
    path_train
    , target_size=(120, 120)
    , batch_size= 32
    , class_mode='binary'
    , color_mode='rgb' # default
    , shuffle='True'
    #Found 20000 images belonging to 2 classes.
)
# print(len(Xy_train[0][0]))
# X_train = Xy_train[0][0]

st = time.time()

# 전체 데이터셋을 하나의 데이터로 만들기
X = []
y = []

for i in range(len(Xy_train)):
    images, labels = Xy_train.next()
    X.append(images)
    y.append(labels)

# all_images와 all_labels을 numpy 배열로 변환하면 하나의 데이터로 만들어진 것입니다.
X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)


# -------------

# X = Xy_train[0][0]
# y = Xy_train[0][1]

# -------------

et = time.time()
print('이미지 변환에 걸린 시간 : ', round(et - st, 3), ' 초')

# print(X.shape)    # (19997, 100, 100, 3)
# print(y.shape)    # (19997,)


model = Sequential()
model.add(Conv2D(9, (3,3) , strides=1, input_shape=(120, 120, 3)))
model.add(Conv2D(64, (2,2), stride=2))
model.add(Dropout(0.4))
model.add(Conv2D(15, (3,3)))
model.add(Flatten())
model.add(Dense(units=8))
model.add(Dense(7, input_shape=(8,)))
model.add(Dense(6))
model.add(Dense(1, activation='sigmoid'))
model.summary()

X_train, X_test, y_train, y_test = train_test_split(X, y ,random_state=1452, train_size=0.86, stratify=y)
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=0, restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])

fit_st = time.time()
model.fit(X_train, y_train, batch_size=32, verbose=1, epochs=1000, validation_split=0.2, callbacks=[es])
fit_et = time.time()

results = model.evaluate(X_test, y_test)
print(results)
print('loss : ' , results[0])
print('acc : ' , results[1])
print("걸린 시간 : ", round(fit_et - fit_st, 3), '초')

file_acc = str(round(results[1], 4))

import datetime
date = datetime.datetime.now().strftime("%m%d_%H%M")
model.save('C:\\_data\\_save\\MCP\\cat_dog\\'+ date + '_' + file_acc +'_cnn.hdf5')

# Xy_test = test_datagen.flow_from_directory(
#     path_test
#     , target_size=(150, 150)
#     , batch_size=120
#     , class_mode='binary'
#     # Found 120 images belonging to 2 classes.
# )


#통배치 -> 이미지 변환에 걸린 시간 :  48.082  초 :
# loss :  0.5302237272262573
# acc :  0.7350000143051147
# 걸린 시간 :  42.12480044364929

# loss :  0.5610533356666565
# acc :  0.715833306312561
# 걸린 시간 :  41.386 초

# loss :  0.48291438817977905
# acc :  0.7741666436195374
# 걸린 시간 :  48.082 초

# 110 batch_size=30 -> 이미지 변환에 걸린 시간 :  20.95  초



