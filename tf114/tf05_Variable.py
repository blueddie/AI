import tensorflow as tf
print(tf.__version__)

sess = tf.compat.v1.Session()

a = tf.Variable([2], dtype=tf.float32)
b = tf.Variable([2], dtype=tf.float32)

init = tf.compat.v1.global_variables_initializer() # 전체 변수 초기화를 해야 한다. 
sess.run(init) 

print(sess.run(a + b))