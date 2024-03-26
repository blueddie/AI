import tensorflow as tf
tf.set_random_seed(777)

# 1. 데이터
x = [1, 2, 3, 4, 5]
y = [3, 5, 7, 9, 11]

# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)  # 정규 분포에서 랜덤한 값
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)  # 정규 분포에서 랜덤한 값

# sess = tf.compat.v1.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(w))    [2.2086694]


# 2. 모델 구성
# y = wx + b    -> 이제는 말할 수 있다. 이거 아님. 나의 세상이 무너졌다..............
hyopthesis = x * w + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hyopthesis - y))    # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)       # == model.compile(loss='mse', optimizer='sgd') 
train = optimizer.minimize(loss)                                        # == model.compile(loss='mse', optimizer='sgd') 


# 3-2. 훈련
# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 변수 초기화

    # model.fit
    epochs = 100
    for step in range(epochs):
        sess.run(train)         # 메인 알고리즘
        if step % 20 == 0:
            print(step + 1, sess.run(loss), sess.run(w), sess.run(b))   # verbose와 model.weight에서 봤던 애들.

    # sess.close() with 문 안에 넣어버리면 sess.close() 하지 않아도 된다. 어차피 with가 종료되면서 sess도 종료된다.
