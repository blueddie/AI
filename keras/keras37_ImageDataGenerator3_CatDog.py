import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D, GlobalAveragePooling2D

train_datagen = ImageDataGenerator(
    rescale=1./255
    , horizontal_flip=True      # 수평 뒤집기
    , vertical_flip=True        # 수직 뒤집기
    , width_shift_range=0.2     # 0-10% 평행 이동하겠다 이동한만큼 0이 채워짐
    , height_shift_range=0.3    # 0-10% 수직  
    , rotation_range=7          # 정해진 각도만큼 이미지를 회전
    , zoom_range=1.3            # 축소 또는 확대
    , shear_range=0.5           # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    , fill_mode='nearest'       # 빈자리를 가장 비슷한 색으로 채움
    , 
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)


path_train = 'c:\\_data\\image\\cat_and_dog\\Train\\'
path_test = 'c:\\_data\\image\\cat_and_dog\\Test\\'

Xy_train = train_datagen.flow_from_directory(
    path_train
    , target_size=(500, 500)
    , batch_size= 20000
    , class_mode='binary'
    , shuffle='True'
    #Found 20000 images belonging to 2 classes.
)
print(len(Xy_train[0][0]))
X_train = Xy_train[0][0]
# for i in range(625):  # 예시로 625번 반복
#     X_train, y_train = Xy_train.next()
#     print(i + 1 , '개 가져옴')


# Xy_test = test_datagen.flow_from_directory(
#     path_test
#     , target_size=(150, 150)
#     , batch_size=120
#     , class_mode='binary'
#     # Found 120 images belonging to 2 classes.
# )