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
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)

# 3-1. 컴파일
# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y)) #  mse
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy

# optimzer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimzer.minimize(loss)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

# [실습]
# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 3000

    for step in range(epochs):
        _, cost_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
        if (step + 1) % 20 == 0:
            print(step + 1, "\t", "loss : ", cost_val)
            
    # predict = tf.nn.softmax(tf.compat.v1.matmul(x, w_val) + b_val)
    # print(sess.run(predict, feed_dict={x:x_data}))
    
    # print(w_val)
#     [[-0.03604688  0.4082428  -0.10867021]
#     [ 0.27579436  0.12929954  0.49501923]
#     [ 0.27735254 -0.29842883 -2.035099  ]
#     [ 0.31695506  0.8199014   2.0081244 ]]
    
    y_predict = sess.run(hypothesis, feed_dict={x:x_data})
    # print(y_predict) # 8행 3열
    # y_predict = sess.run(tf.arg_max(y_predict, 1))
    # print(y_predict)    #[2 0 0 0 2 0 2 2]

    y_predict = np.argmax(y_predict, 1)
    print(y_predict)

    y_data = np.argmax(y_data, axis=1)
    print(y_data)
    from sklearn.metrics import accuracy_score
    acc= accuracy_score(y_predict, y_data)
    print("acc : ", acc)

    # predict = tf.nn.softmax(tf.matmul(x, w_val) + b_val)
    # print(sess.run(predict, feed_dict={x:x_data}))
    # acc = accuracy_score(y, predict)
    # print("acc : ", acc)


    [12, 28, 25]         # 1,202,604.2841647767777492367707679,   1,446,257,064,291.4751736770474229969,   72,004,899,337.385872524161351466126     1,518,263,166,233.1452109779865236998
    [19, 38, 31]         # 178,482,300.96318726084491003378872,   31,855,931,757,113,756.220328671701299,  29,048,849,665,247.42523108568211168
    [21, 42, 34]        # 1,318,815,734.4832146972099988837453,   1,739,274,941,520,501,047.3946813036112, 583,461,742,527,454.88140290273461039

# 7.9209211611744013963850348040238e-7 ,   0.95257337229597734561722857581134,   0.04742583561190653694263178568518