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
                #  , target_size=(150,150)
                 
                 )
print(img)

print(type(img))

plt.imshow(img)
plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape)    # (281, 300, 3) target_size (150, 150, 3)-> (150, 150, 3)
print(type(arr))    # <class 'numpy.ndarray'>

# 차원 증가
img = np.expand_dims(arr, axis=0)
print(img.shape)    # (1, 281, 300, 3)