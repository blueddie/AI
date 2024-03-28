import tensorflow as tf
tf.compat.v1.set_random_seed(3)

# 1. 데이터
x_data = [1, 2, 3, 4, 5]
y_data = [3, 5, 7, 9, 11]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)  # 정규 분포에서 랜덤한 값
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)  # 정규 분포에서 랜덤한 값

hyopthesis = x * w + b

loss = tf.reduce_mean(tf.square(hyopthesis - y))    # mse
lr = 0.001

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0823)       # == model.compile(loss='mse', optimizer='sgd') 
train = optimizer.minimize(loss)       

# [실습]
# 07_2 를 카피해서 아래를 맹글어봐

####################################### 1. Session() // sess.run(변수) ###########################################
# with tf.compat.v1.Session() as sess :
#     sess.run(tf.compat.v1.global_variables_initializer()) # 변수 초기화

#     epochs = 100

#     for step in range(epochs):
        

#         # sess.run(train, feed_dict={x_ph: x, y_ph: y})         # 메인 알고리즘
#         _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})

#         if step % 20 == 0:
#             print(step + 1, loss_val, w_val, b_val)   # verbose와 model.weight에서 봤던 애들.
        
#     # 4. 예측
#     # [실습]
#     x_pred_data = [6,7,8]
#     # 예측값 뽑아봐
#     x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

#     # 1. 파이썬 방식
#     y_pred = x_pred_data * w_val + b_val

#     # 2. placeholder 
#     y_pred2 = x_test * w_val + b_val

#     # print("best_model : ", best_model)
#     print('[6,7,8] 의 예측 : ', sess.run(y_pred2, feed_dict={x_test:x_pred_data}), "w_val : ", w_val, "b_val : " , b_val )

####################################### 2. Session() // 변수.eval(session=sess) ###########################################
# with tf.compat.v1.Session() as sess :
#     sess.run(tf.compat.v1.global_variables_initializer()) # 변수 초기화

#     epochs = 100

#     for step in range(epochs):
        

#         # sess.run(train, feed_dict={x_ph: x, y_ph: y})         # 메인 알고리즘
#         # _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
#         sess.run(train, feed_dict={x:x_data, y:y_data})
#         loss_val = sess.run(loss, feed_dict={x:x_data, y:y_data})
#         w_val = w.eval(session=sess)
#         b_val = b.eval(session=sess)

#         if step % 20 == 0:
#             print(step + 1, loss_val, w_val, b_val)   # verbose와 model.weight에서 봤던 애들.
        
#     # 4. 예측
#     # [실습]
#     x_pred_data = [6,7,8]
#     # 예측값 뽑아봐
#     x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

#     # 1. 파이썬 방식
#     y_pred = x_pred_data * w_val + b_val

#     # 2. placeholder 
#     y_pred2 = x_test * w_val + b_val

#     # print("best_model : ", best_model)
#     print('[6,7,8] 의 예측 : ', y_pred2.eval(session=sess, feed_dict={x_test:x_pred_data}), "w_val : ", w_val, "b_val : " , b_val )

####################################### 3. InteractiveSession() // 변수.eval() ###########################################
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer()) # 변수 초기화

epochs = 100

for step in range(epochs):
    

    # sess.run(train, feed_dict={x_ph: x, y_ph: y})         # 메인 알고리즘
    # _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})

    sess.run(train, feed_dict={x:x_data, y:y_data})
    loss_val = sess.run(loss, feed_dict={x:x_data, y:y_data})
    w_val = w.eval()
    b_val = b.eval()
    if step % 20 == 0:
        print(step + 1, loss_val, w_val, b_val)   # verbose와 model.weight에서 봤던 애들.
    
# 4. 예측
# [실습]
x_pred_data = [6,7,8]
# 예측값 뽑아봐
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

# 1. 파이썬 방식
y_pred = x_pred_data * w_val + b_val

# 2. placeholder 
y_pred2 = x_test * w_val + b_val

# print("best_model : ", best_model)
print('[6,7,8] 의 예측 : ', y_pred2.eval(feed_dict={x_test:x_pred_data}), "w_val : ", w_val, "b_val : " , b_val )
sess.close()