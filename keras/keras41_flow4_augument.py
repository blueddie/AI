from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train / 255.
x_test = x_test / 255.

train_datagen = ImageDataGenerator(
    # rescale=1./255
    horizontal_flip=True
    , vertical_flip=True
    , width_shift_range=0.2
    , height_shift_range=0.2
    , rotation_range=30
    , zoom_range=0.2
    , shear_range=20
    , fill_mode='nearest' # default: nearest
)

augument_size = 40000

randidx = np.random.randint(x_train.shape[0], size=augument_size)
        # np.random.randint(60000, 40000) -> 6만개 중에서 4만개의 숫자를 뽑아라
print(randidx)
print(np.min(randidx), np,max(randidx))

x_augmented = x_train[randidx].copy()   # .copy() -> 별도의 메모리 공간을 할당해서 ?
y_augmented = y_train[randidx].copy()

# print(x_augmented)
# print(x_augmented.shape)    # (40000, 28, 28)
# print(y_augmented)
# print(y_augmented.shape)    # (40000,)

x_augmented = x_augmented.reshape(40000, 28, 28, 1)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented
    , batch_size=augument_size
    , shuffle=False
    ).next()[0]

# print(x_augmented[0])    #(40000, 28, 28, 1)


print(x_train.shape)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape) #(100000, 28, 28, 1) (100000,)
