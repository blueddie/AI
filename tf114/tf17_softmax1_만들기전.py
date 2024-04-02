import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
tf.set_random_seed(777)

# 1. 데이터
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]

y_data = [[0, 0, 1],    # 2
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],    # 1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],    # 0
          [1, 0, 0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
w = tf.compat.v1.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1, 3]), name='bias')

# 2. 모델
hypothesis = tf.compat.v1.matmul(x, w) + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y)) #  mse

# optimzer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimzer.minimize(loss)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)

# [실습]
# 3-2. 훈련
# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 2000

    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
        if (step + 1) % 20:
            print(step + 1, "\t", "loss : ", loss_val)
            

    predict = tf.matmul(x, w_val) + b_val
    print(sess.run(predict, feed_dict={x:x_data}))
    # predict = tf.nn.softmax(tf.matmul(x, w_val) + b_val)
    # print(sess.run(predict, feed_dict={x:x_data}))
    # acc = accuracy_score(y, predict)
    # print("acc : ", acc)
