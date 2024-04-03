import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
import numpy as np
tf.set_random_seed(777)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000, 10)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000, 10)

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28 * 28).astype('float32')
x_test = x_test.reshape(-1, 28 * 28).astype('float32')

# print(x_train.shape, y_train.shape) # (60000, 784) (60000, 10)
# print(x_test.shape, y_test.shape)   # (10000, 784) (10000, 10)

# [실습] 맹그러

# 2. 모델
keep_prob = tf.compat.v1.placeholder(tf.float32)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

# layer1
w1 = tf.compat.v1.get_variable('w1', shape=[784, 128], initializer=tf.contrib.layers.xavier_initializer())  # 초기화는 첫 에포만 진행되고, 그 이후는 비활성화 ★
b1 = tf.compat.v1.Variable(tf.zeros([128]), name='b1')

layer1 = tf.compat.v1.matmul(x , w1) + b1
layer1 = tf.compat.v1.nn.dropout(layer1, rate=0.3)

# layer2
w2 = tf.compat.v1.get_variable('w2', shape=[128, 64], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.compat.v1.Variable(tf.zeros([64]), name='b1')

layer2 = tf.compat.v1.matmul(layer1 , w2) + b2
layer2 = tf.compat.v1.nn.relu(layer2)
layer2 = tf.compat.v1.nn.dropout(layer2, rate=0.3)

# layer3
w3 = tf.compat.v1.get_variable('w3', shape=[64, 32], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.compat.v1.Variable(tf.zeros([32]), name='b3')

layer3 = tf.compat.v1.matmul(layer2 , w3) + b3
layer3 = tf.compat.v1.nn.relu(layer3)
layer3 = tf.compat.v1.nn.dropout(layer3, rate=0.3)

# output
w4 = tf.compat.v1.get_variable('w4', shape=[32, 10], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.compat.v1.Variable(tf.zeros([10]), name='b4')

layer4 = tf.compat.v1.matmul(layer3 , w4) + b4
hypothesis = tf.compat.v1.nn.softmax(layer4)

# 3-1. 컴파일
# loss = tf.compat.v1.losses.softmax_cross_entropy(y, hypothesis)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    epochs = 10000
    for step in range(epochs):
        _, cost_val, w_val, b_val = sess.run([train, loss, w4, b4], feed_dict={x:x_train, y:y_train, keep_prob:0.9})
        if (step + 1) % 20 == 0:
            print(step + 1, "\t", "loss : ", cost_val)
            y_pred = sess.run(hypothesis, feed_dict={x:x_test, keep_prob:1.0})
    y_pred = np.argmax(y_pred, 1)
    y_test = np.argmax(y_test, axis=1)
    acc= accuracy_score(y_test, y_pred)
    print(y_pred)
    print("acc : ", acc)

'''
# layer2 :  model.add(Dense(19))
w1 = tf.compat.v1.Variable(tf.random_normal([784, 19]), name='weight1')
b1 = tf.compat.v1.Variable(tf.zeros([19]), name='bias1')

layer1 = tf.nn.swish(tf.compat.v1.matmul(x, w1) + b1)    # (N, 19)

# layer2 :  model.add(Dense(97))
w2 = tf.compat.v1.Variable(tf.random_normal([19, 97]), name='weight2')
b2 = tf.compat.v1.Variable(tf.zeros([97]), name='bias2')

layer2 = tf.nn.swish(tf.compat.v1.matmul(layer1, w2) + b2)   # (N, 97)
layer2 = tf.nn.dropout(layer2, keep_prob=0.9)

# layer3 :  model.add(Dense(9))
w3 = tf.compat.v1.Variable(tf.random_normal([97, 18]), name='weight3')
b3 = tf.compat.v1.Variable(tf.zeros([18]), name='bias3')

layer3 = tf.nn.swish(tf.compat.v1.matmul(layer2, w3) + b3)   # (N, 9)

# layer4 :  model.add(Dense(21))
w4 = tf.compat.v1.Variable(tf.random_normal([18, 21]), name='weight4')
b4 = tf.compat.v1.Variable(tf.zeros([21]), name='bias4')

layer4 = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(layer3, w4) + b4)   # (N, 21)

# hypothesis : model.add(Dense(1, activation='sigmoid'))
w5 = tf.compat.v1.Variable(tf.random_normal([21, 10]), name='weight5')
b5 = tf.compat.v1.Variable(tf.zeros([10]), name='bias5')

hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(layer4, w5) + b5) # (N, 1)


# 3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
# train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 10000

    for step in range(epochs):
        _, cost_val = sess.run([train, loss], feed_dict={x:x_train, y:y_train, keep_prob:0.9})
        if (step + 1) % 20 == 0:
            print(step + 1, "\t", "loss : ", cost_val)

    y_pred = sess.run(hypothesis, feed_dict={x:x_test, keep_prob:1.0})
    y_pred = np.argmax(y_pred, 1)
    y_test = np.argmax(y_test, axis=1)
    acc= accuracy_score(y_test, y_pred)
    print(y_pred)
    print("acc : ", acc)

# acc :  0.952
'''