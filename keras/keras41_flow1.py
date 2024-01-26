import sys
import tensorflow as tf
# print('tensorflow version : ', tf.__version__)  # 2.9.0
# print('ptthon version : ' , sys.version)   
# 3.9.18 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img  # 이미지를 가져오는 것
from tensorflow.keras.preprocessing.image import img_to_array  # 이미지를 수치화
import numpy as np

path = 'c:\\_data\\image\\cat_and_dog\Train\\Cat\\1.jpg'
img = load_img(path
                 , target_size=(150,150)
                 
                 )
# print(img)

# print(type(img))

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape)    # (281, 300, 3) target_size (150, 150, 3)-> (150, 150, 3)
print(type(arr))    # <class 'numpy.ndarray'>

# 차원 증가
img = np.expand_dims(arr, axis=0)
print(img.shape)    # (1, 281, 300, 3)

################################### 여기부터 증폭 #######################################

datagen = ImageDataGenerator(horizontal_flip=True
                             , vertical_flip=True
                             , width_shift_range=0.2     # 10% 평행 이동하겠다 이동한만큼 0이 채워짐
                             , height_shift_range=0.2    #
                             , rotation_range=5          # 정해진 각도만큼 이미지를 회전
                             , zoom_range=0.2            # 축소 또는 확대
                             , shear_range=20           # 
                             , fill_mode='nearest' )

it = datagen.flow(img
                  , batch_size=1
                  )

flg, ax = plt.subplots(nrows=2, ncols=5, figsize=(10, 10)) # 여러개 사진을  한번에 보겠다


for i in range(2) :
    # print(batch.shape)  #(1, 150, 150, 3)
    # print(image.shape)  #(150, 150, 3)
    for j in range(5):
        batch = it.next()
        image = batch[0].astype('uint8')
        ax[i, j].imshow(image)
        ax[i, j].axis('off')

    # batch = it.next()
    # image = batch[0].astype('uint8')
    # ax[i // 5, i % 5].imshow(image)
    # ax[i // 5, i % 5].axis('off')
    
# print(np.min(batch), np.max(batch))
plt.show() 