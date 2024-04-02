from sklearn.datasets import load_diabetes
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error,accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
tf.random.set_random_seed(777)
np.random.seed(777)
datasets = load_diabetes()

x_data, y_data = datasets.data, datasets.target
print(x_data.shape, y_data.shape)   #(442, 10) (442,)
print(y_data)

y_data = np.reshape(y_data, [-1 ,1])

# scaler= StandardScaler()
scaler= MinMaxScaler()
x = scaler.fit_transform(x_data)
# print(x_data.shape, y_data.shape)   # (442, 10) (442, 1)

# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# layer1 :  model.add(Dense(19))
w1 = tf.compat.v1.Variable(tf.random_normal([10, 19]), name='weight1')
b1 = tf.compat.v1.Variable(tf.zeros([19]), name='bias1')

layer1 = tf.nn.relu(tf.compat.v1.matmul(x, w1) + b1)    # (N, 32)

# layer2 :  model.add(Dense(97))
w2 = tf.compat.v1.Variable(tf.random_normal([19, 97]), name='weight2')
b2 = tf.compat.v1.Variable(tf.zeros([97]), name='bias2')

layer2 = tf.nn.swish(tf.compat.v1.matmul(layer1, w2) + b2)   # (N, 16)

# layer3 :  model.add(Dense(9))
w3 = tf.compat.v1.Variable(tf.random_normal([97, 9]), name='weight3')
b3 = tf.compat.v1.Variable(tf.zeros([9]), name='bias3')

layer3 = tf.nn.swish(tf.compat.v1.matmul(layer2, w3) + b3)   # (N, 16)

# layer4 :  model.add(Dense(21))
w4 = tf.compat.v1.Variable(tf.random_normal([9, 21]), name='weight4')
b4 = tf.compat.v1.Variable(tf.zeros([21]), name='bias4')

layer4 = tf.compat.v1.nn.swish(tf.compat.v1.matmul(layer3, w4) + b4)   # (N, 16)

# hypothesis : model.add(Dense(1, activation='sigmoid'))
w5 = tf.compat.v1.Variable(tf.random_normal([21, 1]), name='weight5')
b5 = tf.compat.v1.Variable(tf.zeros([1]), name='bias5')

hypothesis = tf.compat.v1.matmul(layer4, w5) + b5 # (N, 1)

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)

train = optimizer.minimize(loss)

from sklearn.metrics import mean_squared_error, r2_score

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 50000

    for step in range(epochs):
        _, loss_val= sess.run([train, loss], feed_dict={x:x_data, y:y_data})
        if (step + 1) % 20:
            print(step + 1, "\t", "loss : ", loss_val)

    
    predict = sess.run(hypothesis, feed_dict={x:x_data, y:y_data})
    # predict = sess.run(tf.cast(hypothesis > 0.5 , dtype=tf.float32), feed_dict={x : x_data})
    # acc = accuracy_score(y_data, predict)
    r2 = r2_score(y_data, predict)
    print("r2 : ", r2)

# r2 :  0.7093448408776325