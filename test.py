import tensorflow as tf



init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	a = tf.constant([1,2,3])
	b = tf.placeholder(tf.int32,[3])

	c = tf.add(a,b)
		
	print(sess.run(c,feed_dict = {b : [1,2,3]}))
