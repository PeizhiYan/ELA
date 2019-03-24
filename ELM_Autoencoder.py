"""
This code is a tensorflow implementation of the algorithm on the paper: 
	Autoencoder with Invertible Functions for Dimension Reduction and Image Reconstruction
Authors of the paper:
	Yimin Yang, Jonathan Wu, and Yaonan Wang
"""
##################################################
# Author of this code: Peizhi Yan                #
# Affiliation: Lakehead University               #
# Personal Website: https://PeizhiYan.github.io  #
# Date: March 24th, 2019                         #
# Tensorflow version: r1.13                      #
##################################################

import tensorflow as tf

class ELA:
	"""
	Extreme-Learning Autoencoder (single hidden layer version)
	"""
	def __init__(self, input_units, hidden_units, c, activation='sin'):
		self.input_units = input_units   # number of input neurons
		self.hidden_units = hidden_units # number of hidden neurons
		self.output_units = input_units  # number of output neurons == number of input neurons 
		self.act_flag = activation
		if activation == 'sin':
			self.activation = tf.math.sin
		elif activation == 'sigmoid':
			self.activation = tf.nn.sigmoid
		elif activation == 'identity':
			self.activation = tf.identity
		self.c = c # a constant
		self.X = tf.placeholder(tf.float32, [None, self.input_units]) # input placeholder
		self.ALPHA = tf.get_variable(
			'alpha',
			shape=[self.input_units, self.hidden_units],
			initializer=tf.random_uniform_initializer(-1,1),
			trainable=False
		) # hidden layer (encoding layer) wrights
		self.BETA = tf.get_variable(
			'beta',
			shape=[self.hidden_units, self.input_units],
			initializer=tf.random_uniform_initializer(-1,1),
			trainable=False
		) # output layer (decoding layer) wrights
		self.X_ = tf.matmul(self.activation(tf.matmul(self.X, self.ALPHA)), self.BETA) # prediction
		self.loss = tf.losses.mean_squared_error(labels=self.X, predictions=self.X_) # mean squared loss
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer()) # initialize variables
		self.train_graph = self.build_train_graph()
		self.copy_weight = tf.assign(self.ALPHA, tf.transpose(self.BETA))

	def build_train_graph(self):
		# obtain hidden layer output
		H = self.activation(tf.matmul(self.X, self.ALPHA))

		# compute the Moore-Penrose inverse of H 
		HT = tf.transpose(H)
		I = tf.eye(self.hidden_units) 
		C_I = tf.scalar_mul(self.c, I)
		HTH = tf.matmul(HT, H) 
		K = tf.add(C_I, HTH)
		H_inverse = tf.matmul(tf.matrix_inverse(K), HT)

		# compute decoding layer weights
		inv_activation = None
		if self.act_flag == 'sin':
			arcsin_y = tf.math.asin(self.X)
			inv_activation = arcsin_y
		elif self.act_flag == 'sigmoid':
			# Problem occurs!!!
			#logit = - tf.log(1. / self.X - 1.)
			#inv_activation = logit
			inv_activation = self.X
		elif self.act_flag == 'identity':
			inv_activation = self.X

		BETA_ = tf.assign(self.BETA, tf.matmul(H_inverse, inv_activation)) 

		return BETA_

	def train(self, x, epochs):
		# train the network
		for epoch in range(epochs-1):
			self.sess.run(self.train_graph, feed_dict={self.X: x})
			self.sess.run(self.copy_weight)
		self.sess.run(self.train_graph, feed_dict={self.X: x})

	def evaluate(self, x):
		# get the MSE of the model
		return self.sess.run(self.loss, feed_dict={self.X: x})

	def retrieve_encoding_weights(self):
		# return the weights as a numpy array
		return self.sess.run(self.ALPHA)

	def retrieve_decoding_weights(self):
		# return the weights as a numpy array
		return self.sess.run(self.BETA)

	def encoding(self, x):
		# get the encoding of the input data
		return self.sess.run(self.activation(tf.matmul(self.X, self.ALPHA)), feed_dict={self.X: x})

	def reconstruction(self, x):
		# get the reconstructed data
		return self.sess.run(self.X_, feed_dict={self.X: x})	