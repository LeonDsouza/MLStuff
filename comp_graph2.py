
#TensorBoard - Graphical displays of nodes
import tensorflow as tf

a = tf.placeholder(tf.float32) #placeholder is like declaring float
b = tf.placeholder(tf.float32)
adder_node = a + b  #same as tf.add()


sess = tf.Session()

print(sess.run(adder_node, {a: 3, b:4.5}))  #assign values to a,b
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))  #adds across dimensions


add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b:4.5}))



