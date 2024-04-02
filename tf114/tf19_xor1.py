import tensorflow as tf
from sklearn.metrics import accuracy_score
tf.compat.v1.set_random_seed(777)

# 1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]   # (4, 2)
y_data = [[0], [1], [1], [0]]               # (4, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1]), name='bias')

# 2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)

# 3-1. 컴파일
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 10000

    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
        if (step + 1) % 20:
            print(step + 1, "\t", "loss : ", loss_val)

    predict = sess.run(tf.cast(hypothesis > 0.5 , dtype=tf.float32), feed_dict={x : x_data})
    print(predict)
    acc = accuracy_score(y_data, predict)
    print("acc : ", acc)