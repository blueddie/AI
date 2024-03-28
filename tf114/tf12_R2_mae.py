import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(777)

# 1. 데이터
x_train = [1, 2, 3]
y_train = [1, 2, 3]

x_test = [4,5,6]
y_test = [4,5,6]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

# x_test = tf.compat.v1.placeholder(tf.float32)
# y_test = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')
# b = tf.compat.v1.Variable([10], dtype=tf.float32, name='bias')

# 2. 모델
hypothesis = x * w

# 3-1. 컴파일 // model.compile(loss='mse', optimizer='sgd')

loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse

##################### 옵티마이저 ##############################
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0823)
# train = optimizer.minimize(loss)  
      
lr = 0.1
# gradient = tf.reduce_mean((x * w - y) * x)
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

# print("=================== w history ======================================")
# print(w_hist)
# print("=================== loss history ======================================")
# print(loss_hist)

# plt.plot(loss_hist)
# plt.xlabel("Epochs")
# plt.ylabel('Loss')
# plt.show()

################ [실습] R2, mae 맹그러 !!!!! #################
from sklearn.metrics import r2_score, mean_absolute_error

with tf.compat.v1.Session() as sess:    
    sess.run(tf.compat.v1.global_variables_initializer())         
    
    # 예측값 뽑아봐
    x_pred = tf.compat.v1.placeholder(tf.float32, shape=[None])
    
    # 1. 파이썬 방식
    y_pred = x_pred * w_v

    # 2. placeholder 
    y_pred2 = x_pred * w_v

    # print("best_model : ", best_model)
    ####################################################################
    y_pred = sess.run(y_pred2, feed_dict={x_pred: x_test})
    print('[4,5,6] 의 예측 : ', y_pred)

    r2 = r2_score(y_test, y_pred)
    print("r2 score : ", r2)
    
    mae = mean_absolute_error(y_test, y_pred)
    print("mae : ", mae)