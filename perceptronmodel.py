import tensorflow as tf
import collections


def get_linear_activation(dimension = 10):
	
	def linear_activation(values):

		zeros = tf.zeros([dimension])
		return tf.nn.bias_add(values, zeros, data_format=None, name=None)

	return linear_activation



class Perceptron():

	def __init__(self,input_dim, output_dim,*,activation_function = get_linear_activation()):
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
		return tf.concat(axis = 1,values = outputs)


class Network():

	def __init__(self,*,layer_sizes,dim_of_input):



		assert type(layer_sizes) == list
		assert all((type(size) == int for size in layer_sizes))
		assert type(dim_of_input) == int

		self._create_layers(layer_sizes,dim_of_input)



		#self.x = tf.placeholder(tf.float32, [None, dim_of_input])

	def output(self,data)

		output_of_last_layer = data
		for layer in self.layers:
			output_of_last_layer = layer.output(output_of_last_layer)

		return tf.nn.softmax(output_of_last_layer)

	def evaluation(self,data):
		pass

	def _create_layers(self,layer_sizes,dim_of_input):

		layers = list()

		for previous_layer_size, layer_size in zip([dim_of_input]+layer_sizes[:-1],layer_sizes):
			cell_factory = lambda : Perceptron(previous_layer_size,1,activation_function = tf.sigmoid)

			layer = Layer(	cell_factory 	= cell_factory, 
					number_of_cells	= layer_size)

			layers.append(layer)
		
		self.layers = layers


if __name__ == "__main__":

	Network(layer_sizes 	= [1,2],
		dim_of_input 	= 1)






















