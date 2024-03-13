import tensorflow as tf
tf.set_random_seed(777)

# 1. 데이터
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

# [실습] 맹그러
# 2. 모델 구성
hypothesis = x * w + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(y - hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 100

for step in range(epochs):
    sess.run(train)
    if step % 10 == 0:
        print(step + 1, sess.run(loss), sess.run(w), sess.run(b))

sess.close()    # 세션이 쌓이기 때문에 항상 close