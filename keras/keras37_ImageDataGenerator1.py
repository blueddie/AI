import numpy as np
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255
    , horizontal_flip=True      # 수평 뒤집기
    , vertical_flip=True        # 수직 뒤집기
    , width_shift_range=0.1     # 10% 평행 이동하겠다 이동한만큼 0이 채워짐
    , height_shift_range=0.1    #
    , rotation_range=5          # 정해진 각도만큼 이미지를 회전
    , zoom_range=1.2            # 축소 또는 확대
    , shear_range=0.7           # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    , fill_mode='nearest'       # 빈자리를 가장 비슷한 색으로 채움
    , 
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = 'c:\\_data\\image\\brain\\train\\'
path_test = 'c:\\_data\\image\\brain\\test\\'


Xy_train = train_datagen.flow_from_directory(
    path_train
    , target_size=(200,200)
    , batch_size=160    # 160 이상을 쓰면 X 통으로 가져올 수 있다. batch_size = len(Xy_train)
    , class_mode='binary'
    , color_mode='grayscale'
    , shuffle='True'
    # Found 160 images belonging to 2 classes.
)

Xy_test = test_datagen.flow_from_directory(
    path_test
    , target_size=(200,200)
    , batch_size=120
    , class_mode='binary'
    # , shuffle='True'
    # Found 120 images belonging to 2 classes.
)

print(Xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000001F3E15B4520>
print(Xy_train.next())
print(Xy_train[0])
# print(Xy_train[16]) # error :: 전체데이터 / batch_size = 160/10 16개라 마지막 index는 15다 범위 밖 출력이라 에러

print(Xy_train[0][0]) # 첫번째 배치의 X 
print(Xy_train[0][1]) # 첫번째 배치의 y

print(Xy_train[0][0].shape) #(10, 200, 200, 1) 흑백도 칼라다(o) 칼라는 흑백이다 (X)

print(type(Xy_train))       #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(Xy_train[0]))    #<class 'tuple'>
print(type(Xy_train[0][0])) #<class 'numpy.ndarray'> == X
print(type(Xy_train[0][1])) #<class 'numpy.ndarray'> == y