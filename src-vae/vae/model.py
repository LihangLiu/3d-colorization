import tensorflow as tf
import numpy as np


def weight_variable(name, shape):
	return tf.get_variable(name, shape, tf.float32, tf.random_normal_initializer(stddev = 0.02))

def bias_variable(name, shape):
	return tf.Variable(tf.constant(0.1, shape=shape))
	return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1)),

def addRandomNormal(x):
	return x + tf.random_normal(x.get_shape().as_list())

def conv3d(x, W, stride=2):
	return tf.nn.conv3d(x, W, strides=[1, stride, stride, stride, 1], padding='SAME')

def deconv3d(x, W, output_shape, stride=2):
	return tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, stride, stride, stride, 1], padding='SAME')

def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

class batch_norm(object):
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
			self.epsilon  = epsilon
			self.momentum = momentum
			self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x,
					decay=self.momentum, 
					updates_collections=None,
					epsilon=self.epsilon,
					scale=True,
					is_training=train,
					scope=self.name)

class Generator(object):

	def __init__(self, z_size=20, ngf=9, name="g_"):
		with tf.variable_scope(name):
			self.name = name
			self.ngf = ngf

			self.W = {
				'h1': weight_variable('h1', [4, 4, 4, 1, ngf]),
				'h2': weight_variable('h2', [4, 4, 4, ngf, ngf*2]),
				'h3': weight_variable('h3', [4, 4, 4, ngf*2, ngf*4]),
				'h4': weight_variable('h4', [4, 4, 4, ngf*4, ngf*8]),
				'h5': weight_variable('h5', [4, 4, 4, ngf*8, ngf*16]),

				'hz': weight_variable('hz', [z_size, 2*2*2*ngf*16]),

				'dh1': weight_variable('dh1', [4, 4, 4, ngf*8, ngf*16+ngf*16]),
				'dh2': weight_variable('dh2', [4, 4, 4, ngf*4, ngf*8+ngf*8]),
				'dh3': weight_variable('dh3', [4, 4, 4, ngf*2, ngf*4+ngf*4]),
				'dh4': weight_variable('dh4', [4, 4, 4, ngf, ngf*2+ngf*2]),
				'dh5': weight_variable('dh5', [4, 4, 4, 3, ngf*1+ngf*1])
			}

			self.b = {
				'h1': bias_variable('h1', [ngf]),
				'dh5': bias_variable('dh5', [3]),
			}

			self.bn2 = batch_norm(name = 'bn2')
			self.bn3 = batch_norm(name = 'bn3')
			self.bn4 = batch_norm(name = 'bn4')
			self.bn5 = batch_norm(name = 'bn5')

			self.dbn1 = batch_norm(name = 'dbn1')
			self.dbn2 = batch_norm(name = 'dbn2')
			self.dbn3 = batch_norm(name = 'dbn3')
			self.dbn4 = batch_norm(name = 'dbn4')

	def __call__(self, a, z, train):
		shape = a.get_shape().as_list()		# (n,64,64,64,1)
		ngf = self.ngf

		# conv
		h1 = addRandomNormal(lrelu(conv3d(a,self.W['h1']) + self.b['h1']))	# (n,32,32,32,f)
		h2 = addRandomNormal(lrelu(self.bn2(conv3d(h1,self.W['h2']), train))) # (n,16,16,16,f*2)
		h3 = addRandomNormal(lrelu(self.bn3(conv3d(h2,self.W['h3']), train))) # (n,8,8,8,f*4)
		h4 = addRandomNormal(lrelu(self.bn4(conv3d(h3,self.W['h4']), train))) # (n,4,4,4,f*8)
		h5 = lrelu(self.bn5(conv3d(h4,self.W['h5']), train)) # (n,2,2,2,f*16)

		# add z
		z = tf.matmul(z, self.W['hz'])		# z:(n,2*2*2*f*16)
		z = tf.reshape(z, [-1,2,2,2,ngf*16]) 	# z:(n,2,2,2,f*16)

		# deconv
		z = tf.concat([z,h5], -1)
		dh1 = tf.nn.relu(self.dbn1(deconv3d(z, self.W['dh1'], [shape[0],4,4,4,ngf*8]), train)) #(n,4,4,4,f*8)
		dh1 = tf.concat([dh1,h4], -1)
		dh2 = tf.nn.relu(self.dbn2(deconv3d(dh1, self.W['dh2'], [shape[0],8,8,8,ngf*4]), train)) #(n,8,8,8,f*4)
		dh2 = tf.concat([dh2,h3], -1)
		dh3 = tf.nn.relu(self.dbn3(deconv3d(dh2, self.W['dh3'], [shape[0],16,16,16,ngf*2]), train)) #(n,16,16,16,f*2)
		dh3 = tf.concat([dh3,h2], -1)
		dh4 = tf.nn.relu(self.dbn4(deconv3d(dh3, self.W['dh4'], [shape[0],32,32,32,ngf*1]), train)) #(n,32,32,32,f*1)
		dh4 = tf.concat([dh4,h1], -1)
		rgb = tf.nn.tanh(deconv3d(dh4, self.W['dh5'], [shape[0], 64,64,64,3]) + self.b['dh5']) 	#(n,64,64,64,3)

		# mask
		rgb = mask(rgb,a)
		rgba = tf.concat([rgb,a], -1)

		return rgb, rgba

	# all_z: variable
	def train(self, a, indexes, all_z, train):
		z = tf.gather(all_z, indexes)
		return self.__call__(a, z, train)

	# random_z: placeholder
	def sample(self, a, random_z, train):
		assert tf.get_variable_scope().reuse == False
		return self.__call__(a, random_z, train)

	def fix_shape(self, a, indexes, all_z, train):
		assert tf.get_variable_scope().reuse == False
		N = indexes.get_shape().as_list()[0]
		rep_index = tf.reshape(tf.transpose(tf.reshape(tf.tile(indexes,[N]), (N,-1))),(-1,))
		rep_a = tf.gather(a, rep_index)
		z = tf.gather(all_z, indexes)		
		rep_z = tf.tile(z, [N,1]) 
		return self.__call__(rep_a, rep_z, train)


def mask(rgb, a):
### rgb \in (-1,1), (batch, 32, 32, 32, 3)
### a \in {-1,1},	   (batch, 32, 32, 32, 1)
# (-1,1) -> (0,1)
	rgb = rgb*0.5 + 0.5
	a = a*0.5 + 0.5
	# mask
	rep_a = tf.concat([a,a,a], 4)
	rgb = tf.multiply(rgb,rep_a)
	#rgba = tf.concat([rgb,a], 4)
	# (0,1) -> (-1,1)
	rgb = (rgb-0.5)*2

	return rgb

def get_costMatrix(generated, real, numImages):
	##first reshape generated images and real images
	generated_flat = tf.reshape(generated, [numImages, -1]) 
	real_flat = tf.reshape(real, [numImages, -1]) 
	return pdist2(real_flat, generated_flat)

def pdist2(x1, x2):
	""" Computes the squared Euclidean distance between all pairs """
	C = -2*tf.matmul(x1,tf.transpose(x2))
	nx = tf.reduce_sum(tf.square(x1),1,keep_dims=True)
	ny = tf.reduce_sum(tf.square(x2),1,keep_dims=True)
	costMatrix = (C + tf.transpose(ny)) + nx
			
	return tf.sqrt(1e-3 + costMatrix)

