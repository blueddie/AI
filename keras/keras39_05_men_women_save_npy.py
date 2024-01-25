# https://www.kaggle.com/playlist/men-women-classification
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd

train_datagen = ImageDataGenerator(
    rescale=1./255

)

amplified_train_datagen = ImageDataGenerator(
    rescale=1./255
    , horizontal_flip=True      # 수평 뒤집기
    , vertical_flip=True        # 수직 뒤집기
    , width_shift_range=0.2     # 10% 평행 이동하겠다 이동한만큼 0이 채워짐
    , height_shift_range=0.2    #
    , rotation_range=5          # 정해진 각도만큼 이미지를 회전
    , zoom_range=1.2            # 축소 또는 확대
    , shear_range=0.3           # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    , fill_mode='nearest'  
    
)

submit_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = 'C:\\_data\\kaggle\\men_women\\train\\'
path_amplified_train = 'C:\\_data\\kaggle\\men_women\\train_amplified\\'
path_submit = 'C:\\_data\\kaggle\\men_women\\test\\'



# train image
xy_train = train_datagen.flow_from_directory(
    path_train
    , target_size=(120, 120)
    , batch_size= 32
    , class_mode='binary'
    , color_mode='rgb' # default
    , shuffle='True'
    # Found 3309 images belonging to 2 classes.
)
print('train data ok')

amplified_xytrain = amplified_train_datagen.flow_from_directory(
    path_amplified_train
    , target_size=(120, 120)
    , batch_size= 15
    , class_mode='binary'
    , color_mode='rgb' # default
    , shuffle='True'
     #Found 3309 images belonging to 2 classes
)
print('amplified_train data ok')

# test image
submit = submit_datagen.flow_from_directory(
    path_submit
    , target_size=(120, 120)
    , batch_size= 32
    , class_mode= None
    # Found 1 images belonging to 1 classeS
)
print('submit data ok')

print('사진', submit)


x = []
y = []

for i in range(len(xy_train)) :
    images, labels = xy_train.next()
    # print(images.shape) #(32, 120, 120, 3)
    x.append(images)
    y.append(labels)

x = np.concatenate(x, axis=0)
y = np.concatenate(y, axis=0) 

men_counts = np.sum(y == 0)
women_counts = np.sum(y == 1)

# print(men_counts)   # 1409
# print(women_counts) # 1900

x_amplified = []
y_amplified = []

for i in range(len(amplified_xytrain)) :
    images, labels = xy_train.next()
    
    if women_counts > men_counts :
        idx = np.where(labels == 0)[0]
        if len(idx) > 0 :
            x_amplified.append(images[idx])
            y_amplified.append(labels[idx])
            men_counts += len(idx)
    else :
        break

x_amplified = np.concatenate(x_amplified, axis=0)
y_amplified = np.concatenate(y_amplified, axis=0)
         
x = np.concatenate([x, x_amplified], axis=0)
y = np.concatenate([y, y_amplified], axis=0)    

# men_counts = np.sum(y == 0)
# women_counts = np.sum(y == 1)

# print(men_counts)   #1904
# print(women_counts) #1900

test = []

for i in range(len(submit)) :
    images = submit.next()
    test.append(images)

test = np.concatenate(test, axis=0)

print(x.shape)  #(3807, 120, 120, 3)
print(y.shape)  #(3807,)
# print(test.shape)   #(1, 120, 120, 3)

np_path = 'c:\\_data\\_save_npy\\'
np.save(np_path + 'keras39_5_men_women_x_np.npy', arr=x)
np.save(np_path + 'keras39_5_men_women_y_np.npy', arr=y)
np.save(np_path + 'keras39_5_men_women_test_np.npy', arr=test)


# 데이터 경로
# _data/kaggle/man_woman/
# save load file
# 본인 사진으로 남자인지 여자인지 확인하고 결과치를 메일로 보낸다.

