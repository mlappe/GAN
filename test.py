import tensorflow as tf



init = tf.global_variables_initializer()


class test():

	def __init__(self):
		self.a = tf.constant([1,2,3])
		self.b = tf.placeholder(tf.int32,[3])
		self.c = tf.add(self.a,self.b)

	def __call__(self,session,data = [1,2,3]):
		v = session.run(self.c,feed_dict = {self.b : data})
		return v
		

with tf.Session() as sess:
	sess.run(init)
	print(test()(sess,[1,1,1]))
	a = tf.constant([1,2,3])
	b = tf.placeholder(tf.int32,[3])

	c = tf.add(a,b)
		
	print(sess.run(c,feed_dict = {b : [1,2,3]}))

	

	
