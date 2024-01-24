# https://www.kaggle.com/playlist/men-women-classification
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

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

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = 'C:\\_data\\kaggle\\men_women\\train\\'
path_amplified_train = 'C:\\_data\\kaggle\\men_women\\train_amplified\\'
path_submit = 'C:\\_data\\kaggle\\men_women\\submit\\'

Xy_train = train_datagen.flow_from_directory(
    path_train
    , target_size=(120, 120)
    , batch_size= 32
    , class_mode='binary'
    , color_mode='rgb' # default
    , shuffle='True'
    # Found 3309 images belonging to 2 classes.
)

print('train data ok')

amplified_train = amplified_train_datagen(
    path_amplified_train
    , target_size=(120, 120)
    , batch_size= 32
    , class_mode='binary'
    , color_mode='rgb' # default
    , shuffle='True'
)

print('amplified_train data ok')



# 데이터 경로
# _data/kaggle/man_woman/
# save load file
# 본인 사진으로 남자인지 여자인지 확인하고 결과치를 메일로 보낸다.



