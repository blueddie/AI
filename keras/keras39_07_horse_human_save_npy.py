import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import datetime

# target 300 300 3

path_train = 'C:\\_data\\image\\horse_human\\'

train_datagen = ImageDataGenerator(
    rescale=1./255
    
)

# train2_datagen = ImageDataGenerator(
#     rescale=1.255
#     , horizontal_flip=True      # 수평 뒤집기
#     , vertical_flip=True        # 수직 뒤집기
#     , width_shift_range=0.3     # 10% 평행 이동하겠다 이동한만큼 0이 채워짐
#     , height_shift_range=0.3    #
#     , rotation_range=20          # 정해진 각도만큼 이미지를 회전
#     , zoom_range=1.4            # 축소 또는 확대
#     , shear_range=0.4           # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
#     , fill_mode='nearest'
# ) 


xy_train = train_datagen.flow_from_directory(
    path_train
    , target_size=(300, 300)
    , batch_size=32
    , class_mode='categorical'
    , shuffle=True
    # Found 1027 images belonging to 2 classes.
)

# xy_train2 = train2_datagen.flow_from_directory(
#     path_train
#     , target_size=(300, 300)
#     , batch_size=32
#     , class_mode='categorical'
#     , shuffle=True
# )



x = []
y = []

for i in range(len(xy_train)) :
    images, labels = xy_train.next()
    x.append(images)
    y.append(labels)
    
x = np.concatenate(x, axis=0)
y = np.concatenate(y, axis=0)

# x2 = []
# y2 = []

# for i in range(len(xy_train2)) :
#     images, labels = xy_train2.next()
#     x2.append(images)
#     y2.append(labels)
    
# x2 = np.concatenate(x2, axis=0)
# y2 = np.concatenate(y2, axis=0)

# x = np.concatenate([x, x2])
# y = np.concatenate([y, y2])

# 합치기 전
# print(x.shape)  # (1027, 300, 300, 3)
# print(x.shape)  # (1027, 300, 300, 3)

# #합치기 후
# print(x.shape)  # (2054, 300, 300, 3)
# print(y.shape)  # (2054, 2)

date = datetime.datetime.now().strftime("%m%d_%H%M")



np_path = 'c:\\_data\\_save_npy\\'
np.save(np_path + 'keras39_' + date +'07_horse_human_x_np.npy', arr=x)
np.save(np_path + 'keras39_'+ date + '07_horse_human_y_np.npy', arr=y)