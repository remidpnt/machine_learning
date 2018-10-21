import tensorflow as tf
import numpy as np

arr = np.array(range(9))
arr= arr.reshape((3,3))
print(arr)

itt = tf.placeholder(tf.int32, shape=(3,1))
lookup = tf.nn.embedding_lookup(arr,itt)
hot = tf.one_hot(itt,depth=3)
hot2 = tf.reshape(hot,[3,3])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run( hot2 , feed_dict={itt:np.array(range(3)).reshape((3,1))} ))
