import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(777)

# 1. 데이터

x1_data = [73., 93., 89., 96., 73.]     # 국어
x2_data = [80., 88., 91., 98., 66.]     # 영어
x3_data = [75., 93., 90., 100., 70.]    # 수학
y_data = [152., 185., 180., 196.,142.]  # 환산점수

x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32, name='weight1')
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32, name='weight2')
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32, name='weight3')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32, name='bias')


# 2. 모델
hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y)) # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
# optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)

train = optimizer.minimize(loss)  

# 3-2 훈련
with tf.compat.v1.Session() as sess: 

    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 30000
    for step in range(epochs):
        _, cost_val, w1_v, w2_v, w3_v, b_v = sess.run([train, loss, w1, w2, w3, b], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})
        if (step + 1) % 20 == 0:
            print("epochs:", step + 1, "\t" ,cost_val, "\t", w1_v, "\t", w2_v, "\t", w3_v)

