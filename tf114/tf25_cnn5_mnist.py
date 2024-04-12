import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np
tf.compat.v1.set_random_seed(777)
tf.compat.v1.disable_eager_execution()  # gpu 돌려

# 1. 데이터
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()



from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000, 10)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000, 10)

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.

# 2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1]) # input_shape
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

# Layer1
w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 128],
                            #    initializer=tf.contrib.layers.xavier_initializer()
                               )
                                        # 커널 사이즈, 커널(채널), 필터(아웃풋)
b1 = tf.compat.v1.Variable(tf.zeros([128]), name='b1')

L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID') # 가운데 숫자 2개가 스트라이드 맨 앞 맨 뒤는 shape 맞추는 용 stride 2,2로 주려면 1, 2, 2, 로 바꾸면 된다.
L1 += b1
L1 = tf.nn.relu(L1)
L1 = tf.nn.dropout(L1, rate=0.3)   # //  L1 = tf.nn.dropout(L1, rate=0.3) // model.add(Dropout(0.3))
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

print(L1)   #Tensor("Relu:0", shape=(?, 27, 27, 128), dtype=float32)
print(L1_maxpool)   # Tensor("MaxPool2d:0", shape=(?, 13, 13, 128), dtype=float32)  padding="VALID"
# print(L1_maxpool)   # Tensor("MaxPool2d:0", shape=(?, 14, 14, 128), dtype=float32)  padding="SAME"

# Layer2
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 128, 64],
                            #    initializer=tf.contrib.layers.xavier_initializer()
                               )
                                        # 커널 사이즈, 커널(채널), 필터(아웃풋)
b2 = tf.compat.v1.Variable(tf.zeros([64]), name='b2')

L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1, 1, 1, 1], padding='SAME') # 가운데 숫자 2개가 스트라이드 맨 앞 맨 뒤는 shape 맞추는 용 stride 2,2로 주려면 1, 2, 2, 로 바꾸면 된다.
L2 += b2
L2 = tf.nn.selu(L2)
L2 = tf.nn.dropout(L2, rate=0.1)   # //  L1 = tf.nn.dropout(L1, rate=0.3) // model.add(Dropout(0.3))
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

print(L2)   #Tensor("dropout_1/mul_1:0", shape=(?, 13, 13, 64), dtype=float32)
print(L2_maxpool)   # Tensor("MaxPool2d_1:0", shape=(?, 6, 6, 64), dtype=float32)

# Layer3
w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 64, 32],
                            #    initializer=tf.contrib.layers.xavier_initializer()
                               )
                                        # 커널 사이즈, 커널(채널), 필터(아웃풋)
b3 = tf.compat.v1.Variable(tf.zeros([32]), name='b3')

L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1, 1, 1, 1], padding='SAME') # 가운데 숫자 2개가 스트라이드 맨 앞 맨 뒤는 shape 맞추는 용 stride 2,2로 주려면 1, 2, 2, 로 바꾸면 된다.
L3 += b3
L3 = tf.nn.elu(L3)
# L3 = tf.nn.dropout(L3, keep_prob=0.9)   # //  L1 = tf.nn.dropout(L1, rate=0.3) // model.add(Dropout(0.3))
# L3_maxpool = tf.nn.max_pool2d(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

print(L3)   # Tensor("Elu:0", shape=(?, 6, 6, 32), dtype=float32)

# Flatten
L_flat = tf.reshape(L3, [-1, 6 * 6 * 32])
print(L_flat) # Tensor("Reshape:0", shape=(?, 1152), dtype=float32)

# Layer4 DNN
w4 = tf.compat.v1.get_variable('w4', shape=[6 * 6 * 32, 100],
                                # initializer=tf.contrib.layers.xavier_initializer()
                                )
b4 = tf.compat.v1.Variable(tf.zeros([100], name='b4'))

L4 = tf.nn.relu(tf.matmul(L_flat, w4) + b4)
L4 = tf.nn.dropout(L4, rate=0.3)

# Layer5 DNN
w5 = tf.compat.v1.get_variable('w5', shape=[100, 10],
                                # initializer=tf.contrib.layers.xavier_initializer()
                                )
b5 = tf.compat.v1.Variable(tf.zeros([10], name='b5'))

L5 = tf.nn.relu(tf.matmul(L4, w5) + b5)
hypothesis = tf.nn.softmax(L5)

print(hypothesis)   #Tensor("Softmax:0", shape=(?, 10), dtype=float32)

# 3-1. 컴파일

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.compat.v1.log(hypothesis), axis=1)) # categorical_crossentropy
train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    

    training_epochs = 101
    batch_size = 100 
    total_batch = int(len(x_train) / batch_size) # 60000 / 100

    print(total_batch)  # 600

    for step in range(training_epochs):
        
        avg_cost = 0
        for i in range(total_batch):

            start = i * batch_size
            end = start + batch_size

            batch_x, batch_y = x_train[start:end], y_train[start:end]
            feed_dict = {x:batch_x, y:batch_y}

            _, cost_val, w_val, b_val = sess.run([train, loss, w4, b4], feed_dict=feed_dict)

            avg_cost += cost_val / total_batch


        if (step + 1) % 20 == 0:
            print(step + 1, "\t", "loss : ", avg_cost)
            y_pred = sess.run(hypothesis, feed_dict=feed_dict)

    y_pred = np.argmax(y_pred, 1)
    y_test = np.argmax(y_test, axis=1)
    acc= accuracy_score(y_test, y_pred)
    print(y_pred)
    print("acc : ", acc)