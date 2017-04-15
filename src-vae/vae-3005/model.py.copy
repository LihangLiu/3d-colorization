import tensorflow as tf
import numpy as np


def weight_variable(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.02))

def bias_variable(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

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

	def __init__(self, z_size, name="g_"):
		with tf.variable_scope(name):
			self.name = name
			self.cap = 4
			self.W = {
				'h1': tf.get_variable('h1', [4,4,4,1,self.cap*4], tf.float32, tf.random_normal_initializer(stddev = 0.02)),
				'h2': tf.get_variable('h2', [4,4,4,self.cap*4,self.cap*8], tf.float32, tf.random_normal_initializer(stddev = 0.02)), 
				'h3': tf.get_variable('h3', [4,4,4,self.cap*8,self.cap*16], tf.float32, tf.random_normal_initializer(stddev = 0.02)), 
				'h4': tf.get_variable('h4', [4,4,4,self.cap*16,self.cap*32], tf.float32, tf.random_normal_initializer(stddev = 0.02)), 

				'dh1': tf.get_variable('dh1', [z_size, 2*2*2*self.cap*32], tf.float32, tf.random_normal_initializer(stddev = 0.02)),
				'dh2': tf.get_variable('dh2', [4,4,4,self.cap*16,self.cap*32+self.cap*32], tf.float32, tf.random_normal_initializer(stddev = 0.02)),
				'dh3': tf.get_variable('dh3', [4,4,4,self.cap*8,self.cap*16+self.cap*16], tf.float32, tf.random_normal_initializer(stddev = 0.02)),
				'dh4': tf.get_variable('dh4', [4,4,4,self.cap*4,self.cap*8+self.cap*8], tf.float32, tf.random_normal_initializer(stddev = 0.02)),
				'dh5': tf.get_variable('dh5', [4,4,4,3,self.cap*4+self.cap*4], tf.float32, tf.random_normal_initializer(stddev = 0.02)) 
			}

			self.b = {
				'dh5': tf.get_variable("bias-dh5", [3], initializer=tf.constant_initializer(0.1)),
				'h1': tf.get_variable("bias-h1", [self.cap*4], initializer=tf.constant_initializer(0.1))
			}

			self.bn1 = batch_norm(name = 'bn1')
			self.bn2 = batch_norm(name = 'bn2')
			self.bn3 = batch_norm(name = 'bn3')
			self.bn4 = batch_norm(name = 'bn4')
			self.bn5 = batch_norm(name = 'bn5')
			self.bn6 = batch_norm(name = 'bn6')

	def __call__(self, a, indexes, z, train):
		# shape = z.get_shape().as_list()
		# a = a + tf.random_normal([shape[0], 32, 32, 32, 1])
		z_batch = tf.gather(z, indexes)
		N = indexes.get_shape().as_list()[0]
		h1 = lrelu(conv3d(a,self.W['h1'],stride=2) + self.b['h1']) # (n,16,16,16,8)
		h1 = h1 + tf.random_normal([N, 16, 16, 16, self.cap*4])
		h2 = lrelu(self.bn4(conv3d(h1,self.W['h2'],stride=2),train)) # (n,8,8,8,16)
		h2 = h2 + tf.random_normal([N, 8, 8, 8, self.cap*8])
		h3 = lrelu(self.bn5(conv3d(h2,self.W['h3'],stride=2), train)) # (n,4,4,4,32)
		h3 = h3 + tf.random_normal([N, 4, 4, 4, self.cap*16])

		h4 = lrelu(self.bn6(conv3d(h3,self.W['h4'],stride=2), train)) # (n,2,2,2,64)
		h = tf.nn.relu(tf.matmul(z_batch, self.W['dh1']))
		h = tf.reshape(h, [-1, 2, 2, 2, self.cap*32])

		h = tf.concat([h, h4], -1)
		h = tf.nn.relu(self.bn1(deconv3d(h, self.W['dh2'], [N, 4, 4, 4, self.cap*16]), train))
		h = tf.concat([h, h3], -1)
		h = tf.nn.relu(self.bn2(deconv3d(h, self.W['dh3'], [N, 8, 8, 8, self.cap*8]),train))
		h = tf.concat([h, h2], -1)
		h = tf.nn.relu(self.bn3(deconv3d(h, self.W['dh4'], [N, 16, 16, 16, self.cap*4]), train))
		h = tf.concat([h, h1], -1)
		x = tf.nn.tanh(deconv3d(h, self.W['dh5'], [N, 32, 32, 32, 3]) + self.b['dh5'])
		x = mask(x, a)
		return x

	def fix_shape(self, a, indexes, z, train):
		# with tf.variable_scope('G_z', reuse=True):
		# 	variable_z = tf.get_variable("prior_z", [batch_size,z_size],tf.float32)
		assert tf.get_variable_scope().reuse == False
		variable_z = tf.gather(z, indexes)
		N = indexes.get_shape().as_list()[0]
		
		index_replicate = tf.reshape(tf.transpose(tf.reshape(tf.tile(indexes,[N]), (N,-1))),(-1,))
		a_replicate = tf.gather(a, index_replicate)
		variable_z = tf.tile(variable_z, [N,1])
		h1 = lrelu(conv3d(a_replicate,self.W['h1'],stride=2) + self.b['h1']) # (n,16,16,16,2)
		h2 = lrelu(self.bn4(conv3d(h1,self.W['h2'],stride=2), train)) # (n,8,8,8,4)
		h3 = lrelu(self.bn5(conv3d(h2,self.W['h3'],stride=2), train)) # (n,4,4,4,8)
		h4 = lrelu(self.bn6(conv3d(h3,self.W['h4'],stride=2), train)) # (n,2,2,2,16)
		h = tf.nn.relu(tf.matmul(variable_z, self.W['dh1']))
		h = tf.reshape(h, [-1, 2, 2, 2, self.cap*32])
		h = tf.concat([h, h4], -1)
		h = tf.nn.relu(self.bn1(deconv3d(h, self.W['dh2'], [N**2, 4, 4, 4, self.cap*16]), train))
		h = tf.concat([h, h3], -1)
		h = tf.nn.relu(self.bn2(deconv3d(h, self.W['dh3'], [N**2, 8, 8, 8, self.cap*8]),train))
		h = tf.concat([h, h2], -1)
		h = tf.nn.relu(self.bn3(deconv3d(h, self.W['dh4'], [N**2, 16, 16, 16, self.cap*4]), train))
		h = tf.concat([h, h1], -1)
		x = tf.nn.tanh(deconv3d(h, self.W['dh5'], [N**2, 32, 32, 32, 3]) + self.b['dh5'])

		x = mask(x, a_replicate)

		return x
	def sampler(self, a, z, train):
		#with tf.variable_scope('G_z', reuse=True):
		#	variable_z = tf.get_variable("prior_z", [batch_size,z_size],tf.float32)
		assert tf.get_variable_scope().reuse == False
		N = z.get_shape().as_list()[0]
		h1 = lrelu(conv3d(a,self.W['h1'],stride=2) + self.b['h1']) # (n,16,16,16,8)
		h1 = h1 + tf.random_normal([N, 16, 16, 16, self.cap*4])
		h2 = lrelu(self.bn4(conv3d(h1,self.W['h2'],stride=2), train)) # (n,8,8,8,16)
		h2 = h2 + tf.random_normal([N, 8, 8, 8, self.cap*8])
		h3 = lrelu(self.bn5(conv3d(h2,self.W['h3'],stride=2), train)) # (n,4,4,4,32)
		h3 = h3 + tf.random_normal([N, 4, 4, 4, self.cap*16])

		h4 = lrelu(self.bn6(conv3d(h3,self.W['h4'],stride=2), train)) # (n,2,2,2,64)
		h = tf.nn.relu(tf.matmul(z, self.W['dh1']))
		h = tf.reshape(h, [-1, 2, 2, 2, self.cap*32])

		h = tf.concat([h, h4], -1)
		h = tf.nn.relu(self.bn1(deconv3d(h, self.W['dh2'], [N, 4, 4, 4, self.cap*16]), train))
		h = tf.concat([h, h3], -1)
		h = tf.nn.relu(self.bn2(deconv3d(h, self.W['dh3'], [N, 8, 8, 8, self.cap*8]),train))
		h = tf.concat([h, h2], -1)
		h = tf.nn.relu(self.bn3(deconv3d(h, self.W['dh4'], [N, 16, 16, 16, self.cap*4]), train))
		h = tf.concat([h, h1], -1)
		x = tf.nn.tanh(deconv3d(h, self.W['dh5'], [N, 32, 32, 32, 3]) + self.b['dh5'])
		x = mask(x, a)

		return x

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

