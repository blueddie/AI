from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import numpy as np
tf.compat.v1.set_random_seed(777)

# 1. 데이터
x_data, y_data = load_digits(return_X_y=True)
# print(x_data.shape, y_data.shape) #(1797, 64) (1797,)
y_data = np.reshape(y_data, [-1 ,1])
# print(y_data.shape)
ohe = OneHotEncoder(sparse=False)
y_data = ohe.fit_transform(y_data)
# print(y_data.shape)  #(1797, 10)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=3499)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 64])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
w = tf.compat.v1.Variable(tf.random_normal([64, 10]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1, 10]), name='bias')

# 2. 모델
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)

# 3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1).minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 10000

    for step in range(epochs):
        _, cost_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_train, y:y_train})
        if (step + 1) % 20 == 0:
            print(step + 1, "\t", "loss : ", cost_val)

    y_pred = sess.run(hypothesis, feed_dict={x:x_test})
    y_pred = np.argmax(y_pred, 1)
    y_test = np.argmax(y_test, axis=1)
    acc= accuracy_score(y_test, y_pred)

    print("acc : ", acc)