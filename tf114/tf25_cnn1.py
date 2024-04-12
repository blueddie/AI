import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)

# 1. 데이터
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()



from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000, 10)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000, 10)

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.

# 2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1]) # input_shape
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

# Layer1
w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 64])
                                        # 커널 사이즈, 커널(채널), 필터(아웃풋)

L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID') # 가운데 숫자 2개가 스트라이드 맨 앞 맨 뒤는 shape 맞추는 용 stride 2,2로 주려면 1, 2, 2, 로 바꾸면 된다.
# model.add(Conv2d(64, kernel_size=(2,2), stride=2, input_shape=(28, 28, 1)))

print(w1)   # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1)   # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)

# Layer2
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 64, 32])

L2 = tf.nn.conv2d(L1, w2, strides=[1, 1, 1, 1], padding="VALID")

print(w2)   #<tf.Variable 'w2:0' shape=(3, 3, 64, 32) dtype=float32_ref>
print(L2)   #Tensor("Conv2D_1:0", shape=(?, 25, 25, 32), dtype=float32)

w3 = tf.compat.v1.get_variable('w3', shape=[5, 5, 32, 16])
L3 = tf.nn.conv2d(L2, w3, strides=[1, 1, 1, 1], padding="VALID")

print(w3)   #<tf.Variable 'w3:0' shape=(5, 5, 32, 16) dtype=float32_ref>
print(L3)   #Tensor("Conv2D_2:0", shape=(?, 21, 21, 16), dtype=float32)