import tensorflow as tf
from sklearn.metrics import accuracy_score
tf.compat.v1.set_random_seed(777)

# 1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]   # (4, 2)
y_data = [[0], [1], [1], [0]]               # (4, 1)



# 2. 모델
# layer1 : model.add(Dense(16,input_dim=2))

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random_normal([2, 16]), name='weight1')
b1 = tf.compat.v1.Variable(tf.zeros([16]), name='bias1')

layer1 = tf.nn.relu(tf.compat.v1.matmul(x, w1) + b1)    # (N, 32)

# layer2 :  model.add(Dense(8))
w2 = tf.compat.v1.Variable(tf.random_normal([16, 8]), name='weight2')
b2 = tf.compat.v1.Variable(tf.zeros([8]), name='bias2')

layer2 = tf.nn.relu(tf.compat.v1.matmul(layer1, w2) + b2)   # (N, 16)

# hypothesis : model.add(Dense(1, activation='sigmoid'))
w3 = tf.compat.v1.Variable(tf.random_normal([8, 1]), name='weight3')
b3 = tf.compat.v1.Variable(tf.zeros([1]), name='bias3')

hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer2, w3) + b3) # (N, 1)

# 3-1. 컴파일
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))   # binary_crossentorpy

train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(cost)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(10000):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})

        if step % 200 == 0:
            print(step, cost_val)

    hypo, pred, acc = sess.run([hypothesis, predict, accuracy],
                       feed_dict={x:x_data, y:y_data})
    
    print("훈련 값 : ", hypo)
    print("예측 값 : ", pred)
    print("acc : ", acc)
