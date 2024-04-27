import tensorflow as tf

#   3 + 4 = ?
node1 = tf.constant(3.0, tf.float32)    # 노드 1은 3.0 이라는 상수고 float32형이야 (디폴트 tf.float32)
node2 = tf.constant(4.0)
# node3 = node1 + node2
node3 = tf.add(node1, node2)

print(node1)    #Tensor("Const:0", shape=(), dtype=float32)
print(node2)    #Tensor("Const_1:0", shape=(), dtype=float32)
print(node3)    #Tensor("Add:0", shape=(), dtype=float32)

sess = tf.Session()



print(sess.run(node3))