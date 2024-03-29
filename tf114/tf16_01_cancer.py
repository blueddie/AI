from sklearn.datasets import load_breast_cancer
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error,accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
tf.random.set_random_seed(777)
np.random.seed(777)
datasets = load_breast_cancer()

x, y = datasets.data, datasets.target

# print(x.shape, y.shape) #(569, 30) (569,)

y = np.reshape(y, [-1 ,1])
# print(y.shape)  #(569, 1)
# print(np.unique(y)) #[0 1]

# scaler= StandardScaler()
scaler= MinMaxScaler()
x = scaler.fit_transform(x)


xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])  # x placeholder
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])  # y placeholder

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([30, 1]), dtype=tf.float32, name='weights')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), dtype=tf.float32)

# 2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(xp, w) + b)

# 3-1. 컴파일
loss = -tf.reduce_mean(yp * tf.log(hypothesis) + (1 - yp) * tf.log(1 - hypothesis))   # 'binary_crossentropy'

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)

train = optimizer.minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 1000

    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={xp:x, yp:y})
        if (step + 1) % 20:
            print(step + 1, "\t", "loss : ", loss_val)

    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
    y_pred = tf.sigmoid(tf.matmul(x_test, w_val) + b_val)

    predict = sess.run(tf.cast(y_pred > 0.5 , dtype=tf.float32), feed_dict={x_test : x})
    acc = accuracy_score(y, predict)
    print("acc : ", acc)
