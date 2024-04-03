import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
# 1. 데이터
x_data, y_data = fetch_california_housing(return_X_y=True)

y_data = np.reshape(y_data, [-1, 1])

print(x_data.shape, y_data.shape) #   (20640, 8) (20640,)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=777)

scaler= MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

keep_prob = tf.compat.v1.placeholder(tf.float32)

# layer2 :  model.add(Dense(19))
w1 = tf.compat.v1.Variable(tf.random_normal([8, 19]), name='weight1')
b1 = tf.compat.v1.Variable(tf.zeros([19]), name='bias1')

layer1 = tf.nn.relu(tf.compat.v1.matmul(x, w1) + b1)    # (N, 19)

# layer2 :  model.add(Dense(97))
w2 = tf.compat.v1.Variable(tf.random_normal([19, 97]), name='weight2')
b2 = tf.compat.v1.Variable(tf.zeros([97]), name='bias2')

layer2 = tf.nn.swish(tf.compat.v1.matmul(layer1, w2) + b2)   # (N, 97)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

# layer3 :  model.add(Dense(9))
w3 = tf.compat.v1.Variable(tf.random_normal([97, 9]), name='weight3')
b3 = tf.compat.v1.Variable(tf.zeros([9]), name='bias3')

layer3 = tf.nn.swish(tf.compat.v1.matmul(layer2, w3) + b3)   # (N, 9)

# layer4 :  model.add(Dense(21))
w4 = tf.compat.v1.Variable(tf.random_normal([9, 21]), name='weight4')
b4 = tf.compat.v1.Variable(tf.zeros([21]), name='bias4')

layer4 = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(layer3, w4) + b4)   # (N, 21)

# hypothesis : model.add(Dense(1, activation='sigmoid'))
w5 = tf.compat.v1.Variable(tf.random_normal([21, 1]), name='weight5')
b5 = tf.compat.v1.Variable(tf.zeros([1]), name='bias5')

hypothesis = tf.compat.v1.matmul(layer4, w5) + b5 # (N, 1)

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)


epochs = 10000

with tf.compat.v1.Session() as sess:

    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(epochs):
        _, loss_val = sess.run([train, loss], feed_dict={x:x_train, y:y_train, keep_prob:0.9})

        
        print("epochs:", step + 1, "\t", loss_val)


        
       

        y_pred = sess.run(hypothesis, feed_dict={x: x_test, keep_prob:1.0})

        r2 = r2_score(y_test, y_pred)
        print("r2 score : ", r2)
        
