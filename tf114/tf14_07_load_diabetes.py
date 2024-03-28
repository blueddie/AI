import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터
x, y = load_diabetes(return_X_y=True)
# print(x.shape, y.shape) #(442, 10) (442,)
# y = y.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=777)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None,])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')

# 2. 모델
hypothesis = tf.matmul(xp, w) + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

# 3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101


    
for step in range(epochs):
    _, loss_val, w_v, b_v = sess.run([train, loss, w, b], feed_dict={xp:x_train, yp:y_train})

    
    print("epochs:", step + 1, "\t", loss_val)


x_pred = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y_pred = tf.matmul(x_pred, w_v) + b_v

y_pred = sess.run(y_pred, feed_dict={x_pred: x_test})

r2 = r2_score(y_test, y_pred)
print("r2 score : ", r2)

sess.close()