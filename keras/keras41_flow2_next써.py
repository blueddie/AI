from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255
    , horizontal_flip=True
    , vertical_flip=True
    , width_shift_range=0.2
    , height_shift_range=0.2
    , rotation_range=30
    , zoom_range=0.2
    , shear_range=20
    , fill_mode='nearest' # default: nearest
)

argument_size = 100

# print(x_train[0].shape) #(28, 28)
# plt.imshow(x_train[0])
# plt.show()

x_data = train_datagen.flow(
    
    np.tile(x_train[0].reshape(28*28),argument_size).reshape(-1, 28, 28, 1) # 수치화 된 이미지 데이터 따라서 X
    , np.zeros(argument_size)                                               # 
    , batch_size=argument_size
    , shuffle=True
    ).next()

# print(x_data.shape) 
# 튜플 형태라서 에러 !!! 
# 왜냐하면 flow에서 튜플 형태로 반환했기 때문이다

print(x_data[0].shape) # (100, 28, 28, 1)
print(x_data[1].shape) # (100,)
print(np.unique(x_data[1], return_counts=True))

plt.figure(figsize=(7, 7))
for i in range(49) :
    plt.subplot(7, 7, i + 1)
    plt.axis('off')
    plt.imshow(x_data[0][i], 'gray')
plt.show()

