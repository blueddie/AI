import tensorflow as tf
tf.compat.v1.set_random_seed(337)

# 1. 데이터

x_data = [[73, 51, 65],
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],        # (5,3)
          [17, 66, 79]]

y_data = [[152], [185], [180], [205], [142]] # (5, 1)


################ [실습] ########################
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random_normal([3, 1], name='weight'))
# w = tf.compat.v1.Variable(tf.random_normal([1], name='weight'))
b = tf.compat.v1.Variable(tf.random_normal([1], name='bias'))

# 2. 모델
hypothesis = tf.matmul(x, w) + b
# hypothesis = x * w + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y)) # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
# optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)

train = optimizer.minimize(loss)


# 3-2 훈련
with tf.compat.v1.Session() as sess: 

    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 2000
    for step in range(epochs):
        _, loss_val = sess.run([train, loss], feed_dict={x:x_data, y:y_data})
        if (step + 1) % 20 == 0:
            print("epochs:", step + 1, "\t", loss_val)