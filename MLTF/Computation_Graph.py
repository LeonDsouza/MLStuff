import tensorflow as tf 

#create two floating point tensors
#constant tensors dont take any input or give any output
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)


sess = tf.Session()
print(sess.run([node1, node2])) #[3.0, 4.0]




####################
node3 = tf.add(node1, node2) #adds values of node1 and node 2
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))

###################

