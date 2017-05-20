import tensorflow as tf
import collections


def get_linear_activation(dimension = 10):
	
	def linear_activation(values):

		zeros = tf.zeros([dimension])
		return tf.nn.bias_add(values, zeros, data_format=None, name=None)

	return linear_activation



class Perceptron():

	def __init__(self,input_dim, output_dim,*,activation_function = None):
		"""
		for example 
				activation_function = tf.sigmoid
		"""

		activation_function = get_linear_activation(output_dim) if activation_function == None else activation_function

		self.weights = tf.Variable(tf.zeros([input_dim, output_dim]))
		self.bias = tf.Variable(tf.random_normal([output_dim]))

		self.activation_function = activation_function




	def output(self,data):

		return self.activation_function(tf.matmul(data, self.weights) + self.bias)


class Layer():
	def __init__(self,*,cell_factory, number_of_cells):
		self.cells = [cell_factory() for i in range(number_of_cells)]

	def output(self,data):
		outputs = [cell.output(data) for cell in self.cells]

		print("layer.out",outputs)
		return tf.concat(axis = 1,values = outputs)


class Network():

	def __init__(self,*,layer_sizes,dim_of_input):



		assert type(layer_sizes) == list
		assert all((type(size) == int for size in layer_sizes))
		assert type(dim_of_input) == int

		dim_of_output = layer_sizes[-1]

		self._create_layers(layer_sizes,dim_of_input)



		self.x 		= tf.placeholder(tf.float32, [None, dim_of_input], name = "data")
		self.true_y 	= tf.placeholder(tf.float32, [None, dim_of_output])
		self.y 		= self.output(self.x)

		self.learning_rate = tf.placeholder(tf.float32, shape=[])
		#self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.true_y * tf.log(self.y), reduction_indices=[1]))

		print("self.true_y",self.true_y.get_shape().as_list())
		print("self.y",self.y.get_shape().as_list())
		
		self.loss = tf.losses.mean_squared_error(labels = self.true_y,predictions = self.y)

		self.train_step = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

	def output(self,data):

		output_of_last_layer = data
		for layer in self.layers:
			output_of_last_layer = layer.output(output_of_last_layer)

		print("out_last",output_of_last_layer.get_shape().as_list())

		return output_of_last_layer

		return tf.nn.softmax(output_of_last_layer)

	def evaluate(self,data,tf_session):


		return tf_session.run(self.y, feed_dict={self.x : data})


	def train(self,data,labels,learning_rate,tf_session):

		tf_session.run(self.train_step, feed_dict={self.x: data, self.true_y: labels, self.learning_rate: learning_rate})

		#correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.true_y,1))

		#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		#testaccuracy =  sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
		#trainaccuracy = sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})

		#yield Result(i,testaccuracy,trainaccuracy)


	def _create_layers(self,layer_sizes,dim_of_input):

		layers = list()

		for previous_layer_size, layer_size in zip([dim_of_input]+layer_sizes[:-1],layer_sizes):
			cell_factory = lambda : Perceptron(previous_layer_size,1,activation_function = tf.sigmoid)

			layer = Layer(	cell_factory 	= cell_factory, 
					number_of_cells	= layer_size)

			layers.append(layer)
		
		self.layers = layers


if __name__ == "__main__":

	network = Network(	layer_sizes 	= [1,2],
				dim_of_input 	= 1)


	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		v = network.evaluate([[1],[2],[1]],sess)
		print(v)


	network = Network(	layer_sizes 	= [1],
				dim_of_input 	= 1)


	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		data = [[1],[2],[3]]
		labels = [[0],[0],[1]]
		v = network.evaluate(data,sess)
		print(v)

		for i in range(5):

			network.train(data,labels,0.5,sess)


			v = network.evaluate(data,sess)
			print(v)







		

















