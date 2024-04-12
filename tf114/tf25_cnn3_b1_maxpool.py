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
w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 128])
                                        # 커널 사이즈, 커널(채널), 필터(아웃풋)
b1 = tf.compat.v1.Variable(tf.zeros([128]), name='b1')

L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID') # 가운데 숫자 2개가 스트라이드 맨 앞 맨 뒤는 shape 맞추는 용 stride 2,2로 주려면 1, 2, 2, 로 바꾸면 된다.
L1 += b1
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

print(L1)   #Tensor("Relu:0", shape=(?, 27, 27, 128), dtype=float32)
print(L1_maxpool)   # Tensor("MaxPool2d:0", shape=(?, 13, 13, 128), dtype=float32)  padding="VALID"
# print(L1_maxpool)   # Tensor("MaxPool2d:0", shape=(?, 14, 14, 128), dtype=float32)  padding="SAME"
'''
print(w1)   # <tf.Variable 'w1:0' shape=(2, 2, 1, 128) dtype=float32_ref>
print(L1)   # Tensor("Relu:0", shape=(?, 27, 27, 128), dtype=float32)

# Layer2
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 128, 64])
b1 = tf.compat.v1.Variable(tf.zeros([64]), name='b1')
L2 = tf.nn.conv2d(L1, w2, strides=[1, 1, 1, 1], padding="VALID")

print(w2)   #<tf.Variable 'w2:0' shape=(3, 3, 64, 32) dtype=float32_ref>
print(L2)   #Tensor("Conv2D_1:0", shape=(?, 25, 25, 32), dtype=float32)

# Layer3
w3 = tf.compat.v1.get_variable('w3', shape=[5, 5, 32, 16])
L3 = tf.nn.conv2d(L2, w3, strides=[1, 1, 1, 1], padding="VALID")

print(w3)   #<tf.Variable 'w3:0' shape=(5, 5, 32, 16) dtype=float32_ref>
print(L3)   #Tensor("Conv2D_2:0", shape=(?, 21, 21, 16), dtype=float32)

'''