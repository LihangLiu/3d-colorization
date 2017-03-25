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

def l1_loss(weights, scale, scope=None):
	l1_regularizer = tf.contrib.layers.l1_regularizer(scale=scale, scope=scope)
	loss = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
	return loss


class BatchNormalization(object):

	def __init__(self, shape, name, decay=0.9, epsilon=1e-5):
		with tf.variable_scope(name):
			self.beta = tf.Variable(tf.constant(0.0, shape=shape), name="beta") # offset
			self.gamma = tf.Variable(tf.constant(1.0, shape=shape), name="gamma") # scale
			self.ema = tf.train.ExponentialMovingAverage(decay=decay)
			self.epsilon = epsilon

	def __call__(self, x, train):
		self.train = train
		n_axes = len(x.get_shape()) - 1
		batch_mean, batch_var = tf.nn.moments(x, range(n_axes))
		mean, variance = self.ema_mean_variance(batch_mean, batch_var)
		return tf.nn.batch_normalization(x, mean, variance, self.beta, self.gamma, self.epsilon)

	def ema_mean_variance(self, mean, variance):
		def with_update():
			ema_apply = self.ema.apply([mean, variance])
			with tf.control_dependencies([ema_apply]):
				return tf.identity(mean), tf.identity(variance)
		return tf.cond(self.train, with_update, lambda: (self.ema.average(mean), self.ema.average(variance)))

# code from https://github.com/openai/improved-gan
class VirtualBatchNormalization(object):

	def __init__(self, x, name, epsilon=1e-5, half=None):
		"""
		x is the reference batch
		"""
		assert isinstance(epsilon, float)

		self.half = half
		shape = x.get_shape().as_list()
		needs_reshape = len(shape) != 4

		if needs_reshape:
			orig_shape = shape
			if len(shape) == 5:
				x = tf.reshape(x, [shape[0], 1, shape[1]*shape[2]*shape[3], shape[4]])
			elif len(shape) == 2:
				x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
			elif len(shape) == 1:
				x = tf.reshape(x, [shape[0], 1, 1, 1])
			else:
				assert False, shape
			shape = x.get_shape().as_list()

		with tf.variable_scope(name) as scope:
			assert name.startswith("d_") or name.startswith("g_")
			self.epsilon = epsilon
			self.name = name
			if self.half is None:
				half = x
			elif self.half == 1:
				half = tf.slice(x, [0, 0, 0, 0], [shape[0] // 2, shape[1], shape[2], shape[3]])
			elif self.half == 2:
				half = tf.slice(x, [shape[0] // 2, 0, 0, 0], [shape[0] // 2, shape[1], shape[2], shape[3]])
			else:
				assert False
			self.mean = tf.reduce_mean(half, [0, 1, 2], keep_dims=True)
			self.mean_sq = tf.reduce_mean(tf.square(half), [0, 1, 2], keep_dims=True)
			self.batch_size = int(half.get_shape()[0])
			assert x is not None
			assert self.mean is not None
			assert self.mean_sq is not None
			out = self._normalize(x, self.mean, self.mean_sq, "reference")
			if needs_reshape:
				out = tf.reshape(out, orig_shape)
			self.reference_output = out

	def __call__(self, x):
		shape = x.get_shape().as_list()
		needs_reshape = len(shape) != 4

		if needs_reshape:
			orig_shape = shape
			if len(shape) == 5:
				x = tf.reshape(x, [shape[0], 1, shape[1]*shape[2]*shape[3], shape[4]])
			elif len(shape) == 2:
				x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
			elif len(shape) == 1:
				x = tf.reshape(x, [shape[0], 1, 1, 1])
			else:
				assert False, shape
			shape = x.get_shape().as_list()

		with tf.variable_scope(self.name, reuse=True) as scope:
			new_coeff = 1. / (self.batch_size + 1.)
			old_coeff = 1. - new_coeff
			new_mean = tf.reduce_mean(x, [0, 1, 2], keep_dims=True)
			new_mean_sq = tf.reduce_mean(tf.square(x), [0, 1, 2], keep_dims=True)
			mean = new_coeff * new_mean + old_coeff * self.mean
			mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
			out = self._normalize(x, mean, mean_sq, "live")
			if needs_reshape:
				out = tf.reshape(out, orig_shape)
			return out

	def _normalize(self, x, mean, mean_sq, message):
		# make sure this is called with a variable scope
		shape = x.get_shape().as_list()
		assert len(shape) == 4
		self.gamma = tf.get_variable("gamma", [shape[-1]], initializer=tf.random_normal_initializer(1., 0.02))
		self.beta = tf.get_variable("beta", [shape[-1]], initializer=tf.constant_initializer(0.))
		gamma = tf.reshape(self.gamma, [1, 1, 1, -1])
		beta = tf.reshape(self.beta, [1, 1, 1, -1])
		assert self.epsilon is not None
		assert mean_sq is not None
		assert mean is not None
		std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))
		out = x - mean
		out = out / std
		# out = tf.Print(out, [tf.reduce_mean(out, [0, 1, 2]),
		#	tf.reduce_mean(tf.square(out - tf.reduce_mean(out, [0, 1, 2], keep_dims=True)), [0, 1, 2])],
		#	message, first_n=-1)
		out = out * gamma
		out = out + beta
		return out

def vbn(x, name):
	f = VirtualBatchNormalization(x, name)
	return f(x)


### deeper G
### 
class Generator(object):

	def __init__(self, z_size=5, ngf=8, name="g_"):
		with tf.variable_scope(name):
			self.name = name
			self.ngf = ngf

			self.W = {
				'hz': weight_variable([z_size, 32*32*32*1]),

				'h1': weight_variable([4, 4, 4, 2, ngf]),
				'h2': weight_variable([4, 4, 4, ngf, ngf*2]),
				'h3': weight_variable([4, 4, 4, ngf*2, ngf*4]),
				'h4': weight_variable([4, 4, 4, ngf*4, ngf*8]),
				'h5': weight_variable([4, 4, 4, ngf*8, ngf*8]),

				'dh1': weight_variable([4, 4, 4, ngf*8, ngf*8]),
				'dh2': weight_variable([4, 4, 4, ngf*4, ngf*8]),
				'dh3': weight_variable([4, 4, 4, ngf*2, ngf*4]),
				'dh4': weight_variable([4, 4, 4, ngf, ngf*2]),
				'dh5': weight_variable([4, 4, 4, 3, ngf])
			}

			self.b = {
				'h1': bias_variable([ngf]),
				'dh5': bias_variable([3]),
			}

			self.bn2 = BatchNormalization([ngf*2], 'bn2')
			self.bn3 = BatchNormalization([ngf*4], 'bn3')
			self.bn4 = BatchNormalization([ngf*8], 'bn4')
			self.bn5 = BatchNormalization([ngf*8], 'bn5')

	def __call__(self, a, z, train):
		shape = a.get_shape().as_list()		# (n,32,32,32,1)
		ngf = self.ngf

		# add noise
		z = tf.matmul(z, self.W['hz'])		# z:(n,32*32*32)
		z = tf.reshape(z, [-1,32,32,32,1]) 	# z:(n,32,32,32,1)
		h = tf.concat([a,z], -1)			# (n,32,32,32,2)

		# conv
		h1 = lrelu(conv3d(h,self.W['h1'],stride=2) + self.b['h1'])	# (n,16,16,16,f)
		h2 = lrelu(self.bn2(conv3d(h1,self.W['h2'],stride=2), train)) # (n,8,8,8,f*2)
		h3 = lrelu(self.bn3(conv3d(h2,self.W['h3'],stride=2), train)) # (n,4,4,4,f*4)
		h4 = lrelu(self.bn4(conv3d(h3,self.W['h4'],stride=2), train)) # (n,2,2,2,f*8)
		h5 = lrelu(self.bn5(conv3d(h4,self.W['h5'],stride=2), train)) # (n,1,1,1,f*8)

		# deconv
		dh1 = tf.nn.relu(vbn(deconv3d(h5, self.W['dh1'], [shape[0],  2,2,2,ngf*8]), 'g_vbn_1')) #(n,2,2,2,f*8)
		dh2 = tf.nn.relu(vbn(deconv3d(dh1, self.W['dh2'], [shape[0], 4,4,4,ngf*4]), 'g_vbn_2')) #(n,4,4,4,f*4)
		dh3 = tf.nn.relu(vbn(deconv3d(dh2, self.W['dh3'], [shape[0], 8,8,8,ngf*2]), 'g_vbn_3')) #(n,8,8,8,f*2)
		dh4 = tf.nn.relu(vbn(deconv3d(dh3, self.W['dh4'], [shape[0], 16,16,16,ngf]), 'g_vbn_4')) #(n,16,16,16,f)
		rgb = tf.nn.tanh(deconv3d(dh4, self.W['dh5'], [shape[0], 32,32,32,3]) + self.b['dh5']) 	#(n,32,32,32,3)

		# mask
		rgba = self.mask(rgb,a)

		return rgba

	def get_spaconv_loss(self, scale):
		weights = [self.W['h1'],self.W['h2'],self.W['h3'],
					self.W['h4'],self.W['h5'],self.W['dh1'],
					self.W['dh2'],self.W['dh3'],self.W['dh4']]
		return l1_loss(weights, scale, scope=self.name)

	def mask(self, rgb, a):
		### rgb \in (-1,1), (batch, 32, 32, 32, 3)
		### a \in {-1,1},	(batch, 32, 32, 32, 1)
		# (-1,1) -> (0,1)
		rgb = rgb*0.5 + 0.5
		a = a*0.5 + 0.5
		# mask
		rep_a = tf.concat([a,a,a], 4)
		rgb = tf.multiply(rgb,rep_a)
		rgba = tf.concat([rgb,a], 4)
		# (0,1) -> (-1,1)
		rgba = (rgba-0.5)*2

		return rgba


class Discriminator(object):

	def __init__(self, name="d_"):
		with tf.variable_scope(name):
			self.name = name

			self.W = {
				'h1': weight_variable([4, 4, 4, 4, 16]),
				'h2': weight_variable([4, 4, 4, 16, 32]),
				'h3': weight_variable([4, 4, 4, 32, 64]),
				'h4': weight_variable([4, 4, 4, 64, 128]),
				'h5': weight_variable([2*2*2*128, 2]),
			}

			self.b = {
				'h1': bias_variable([16]),
				'h5': bias_variable([2]),
			}

			self.bn2 = BatchNormalization([32], 'bn2')
			self.bn3 = BatchNormalization([64], 'bn3')
			self.bn4 = BatchNormalization([128], 'bn4')

	def __call__(self, x, train):
		shape = x.get_shape().as_list()		
		noisy_x = x + tf.random_normal(shape,mean=0.0,stddev=1)
		
		h = lrelu(conv3d(noisy_x, self.W['h1']) + self.b['h1'])
		h = lrelu(self.bn2(conv3d(h, self.W['h2']), train))
		h = lrelu(self.bn3(conv3d(h, self.W['h3']), train))
		h = lrelu(self.bn4(conv3d(h, self.W['h4']), train))
		h = tf.reshape(h, [-1, 2*2*2*128])

		y = tf.matmul(h, self.W['h5']) + self.b['h5']
		return y

	def get_spaconv_loss(self, scale):
		weights = [self.W['h1'],self.W['h2'],self.W['h3'],self.W['h4']]
		return l1_loss(weights, scale, scope=self.name)



