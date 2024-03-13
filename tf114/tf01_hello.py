import tensorflow as tf
print(tf.__version__)   # 1.14.0

print("hello world")

hello = tf.constant('hello world')
print(hello)    # Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session()
print(sess.run(hello))