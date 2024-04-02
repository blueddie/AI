import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
tf.compat.v1.set_random_seed(777)

# 1. 데이터
x_data, y_data = fetch_covtype(return_X_y=True)
print(x_data.shape, y_data.shape)   # (1797, 64) (1797,)

y_data = np.reshape(y_data, [-1, 1])
# print(np.unique(y_data))
ohe = OneHotEncoder(sparse=False)
y_data = ohe.fit_transform(y_data)

# print(x_data.shape, y_data.shape)       #(581012, 54) (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=3499)

# 2. 모델
# layer1 : model.add(Dense(16,input_dim=10))
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 54])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 7])

w1 = tf.compat.v1.Variable(tf.random_normal([54, 32]), name='weight1')
b1 = tf.compat.v1.Variable(tf.zeros([32]), name='bias1')

layer1 = tf.nn.swish(tf.compat.v1.matmul(x, w1) + b1)    # (N, 32)

# layer2 :  model.add(Dense(8))
w2 = tf.compat.v1.Variable(tf.random_normal([32, 16]), name='weight2')
b2 = tf.compat.v1.Variable(tf.zeros([16]), name='bias2')

layer2 = tf.nn.sigmoid(tf.compat.v1.matmul(layer1, w2) + b2)   # (N, 16)

# hypothesis : model.add(Dense(1, activation='sigmoid'))
w3 = tf.compat.v1.Variable(tf.random_normal([16, 7]), name='weight3')
b3 = tf.compat.v1.Variable(tf.zeros([7]), name='bias3')

hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer2, w3) + b3) # (N, 1)

# 3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)


# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 100

    for step in range(epochs):
        _, cost_val = sess.run([train, loss], feed_dict={x:x_train, y:y_train})
        if (step + 1) % 20 == 0:
            print(step + 1, "\t", "loss : ", cost_val)

    y_pred = sess.run(hypothesis, feed_dict={x:x_test})
    y_pred = np.argmax(y_pred, 1)
    y_test = np.argmax(y_test, axis=1)
    acc= accuracy_score(y_test, y_pred)
    print(y_pred)
    print("acc : ", acc)
