import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(777)

# 1. 데이터
x_train = [1, 2, 3]
y_train = [1, 2, 3]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')

# 2. 모델
hypothesis = x * w

# 3-1. 컴파일 // model.compile(loss='mse', optimizer='sgd')

loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse

##################### 옵티마이저 ##############################
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0823)
# train = optimizer.minimize(loss)  
      
lr = 0.1
gradient = tf.reduce_mean((x * w - y) * x)

descent = w - lr * gradient

update = w.assign(descent)
##################### 옵티마이저 ##############################

w_hist = []
loss_hist = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(31) :
    
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={x:x_train, y:y_train})
    print(step, '\t', loss_v, '\t', w_v)
    w_hist.append(w_v)
    loss_hist.append(loss_v)

sess.close()

print("=================== w history ======================================")
print(w_hist)
print("=================== loss history ======================================")
print(loss_hist)

plt.plot(loss_hist)
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.show()
