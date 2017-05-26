import tensorflow as tf
import collections
import logging
from collections import namedtuple

logger = logging.getLogger(__name__)


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
	
	@property
	def theta(self):
		"""Returns model parameters
		"""
		return [self.weights,self.bias]


class Layer():
	def __init__(self,*,cell_factory, number_of_cells):
		self.cells = [cell_factory() for i in range(number_of_cells)]

	def output(self,data):
		outputs = [cell.output(data) for cell in self.cells]

		logger.debug("layer.out" + str(outputs))
		return tf.concat(axis = 1,values = outputs)


	@property
	def theta(self):
		"""Returns model parameters
		"""
		cells = [cell.theta for cell in self.cells]
		theta = [parameter for cell in cells for parameter in cell]
		return theta


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

		logger.debug("self.true_y"+str(self.true_y.get_shape().as_list()))
		logger.debug("self.y"+str(self.y.get_shape().as_list()))
		
		self.loss = tf.losses.mean_squared_error(labels = self.true_y,predictions = self.y)

		self.train_step = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

	@property
	def theta(self):
		"""Returns model parameters
		"""
		layer_parameters = [layer.theta for layer in self.layers]
		theta = [parameter for layer in layer_parameters for parameter in layer]
		return theta

	def output(self,data):

		output_of_last_layer = data
		for layer in self.layers:
			output_of_last_layer = layer.output(output_of_last_layer)

		logger.debug("out_last"+str(output_of_last_layer.get_shape().as_list()))

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


class GAN():

	def __init__(self,generator,discriminator):
		
		self.generator 		= generator
		self.discriminator 	= discriminator

	def generate(self,data,tf_session):
		pass

	def train(self,random_seeds,data,learning_rate,tf_session):
		pass


if __name__ == "__main__":

	logging.basicConfig(level=logging.INFO)

	from hypothesis import given
	import hypothesis.strategies as strategy

	import numpy

	import argparse
	parser = argparse.ArgumentParser()

	parser.add_argument("--test", help="run tests", action="store_true")

	args = parser.parse_args()

	run_tests = args.test

	logger.info(str(args))


	Testnetwork = namedtuple("Testnetwork",["Network","dim_of_input","layer_sizes","test_input"])

	def test_network_setup(data):
		"""setup for test
		returns a network with variable size (actualy a namedtuple) 
		data is hypothesis.strategy.data()
		"""


		dim_of_input 		= data.draw(strategy.integers(min_value=1, max_value=10))

		possible_layersizes	= strategy.integers(min_value=1, max_value=10)
		layer_sizes 		= data.draw(strategy.lists(elements=possible_layersizes, min_size=1, average_size=None, max_size=5, unique_by=None, unique=False))

		number_of_inputs 	= data.draw(strategy.integers(min_value=1, max_value=10))

		possible_input_fvalues	= strategy.integers(min_value=1, max_value=10)
		possible_input_vector	= data.draw(strategy.lists(	elements = possible_input_fvalues, 
									min_size = dim_of_input, 
									max_size = dim_of_input))
		input_for_evaluate 	= [possible_input_vector for i in range(number_of_inputs) ]

		network = Network(	layer_sizes 	= layer_sizes,
					dim_of_input 	= dim_of_input)

		return Testnetwork(network,dim_of_input,layer_sizes ,input_for_evaluate)





	@given(strategy.data())
	def test_network1(data):
		"""
		Testing format of output of evaluation
		no semantic checking
		"""
		pass

		"""definition of input values"""

		network,dim_of_input,layer_sizes ,input_for_evaluate = test_network_setup(data)
		number_of_inputs = len(input_for_evaluate)


		"""actual test"""


		init = tf.global_variables_initializer()

		with tf.Session() as sess:
			sess.run(init)
			result = network.evaluate(input_for_evaluate,sess)

		assert len(result) == number_of_inputs 
		assert all([len(sample_features) == layer_sizes[-1] for sample_features in result])

		is_float = [type(value) == numpy.float32 for sample in result for value in sample]
		assert all(is_float)
		logger.debug("test passed")

	


	def test_network2():
		"""
		Test whether a single Perceptron learns to learn a  decision boundary between 2 and 3
		for 1d inputs
		"""
		network = Network(	layer_sizes 	= [1],
					dim_of_input 	= 1)


		init = tf.global_variables_initializer()

		with tf.Session() as sess:
			sess.run(init)
			data = [[1],[2],[3]]
			labels = [[0],[0],[1]]
			v = network.evaluate(data,sess)

			for i in range(5000):

				network.train(data,labels,0.5,sess)


			v = network.evaluate(data,sess)
			assert v[2] > 0.9
			assert v[1] < 0.15
			assert v[0] < 0.15



	if run_tests:
		logger.info("running tests")
		test_network1()
		test_network2()
		logger.info("all tests successfull")
	else:
		logger.info("skipping tests")


	network = Network(	layer_sizes 	= [1,2],
					dim_of_input 	= 1)

	print(network.theta)


		

















