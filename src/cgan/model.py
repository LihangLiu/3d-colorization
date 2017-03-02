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

class Generator(object):

	def __init__(self, z_size, name="g_"):
		with tf.variable_scope(name):
			self.name = name
			# weights for decoder
			self.d_W = {
				'h1': weight_variable([z_size+2*2*2*128, 2*2*2*128]),   #!!!tbd
				'h2': weight_variable([4, 4, 4, 64, 128]),
				'h3': weight_variable([4, 4, 4, 32, 64]),
				'h4': weight_variable([4, 4, 4, 16, 32]),
				'h5': weight_variable([4, 4, 4, 3, 16])
			}

			self.d_b = {
				'h5': bias_variable([1])
			}
			# weights for encoder
			self.e_W = {
				'h1': weight_variable([4, 4, 4, 1, 16]),
				'h2': weight_variable([4, 4, 4, 16, 32]),
				'h3': weight_variable([4, 4, 4, 32, 64]),
				'h4': weight_variable([4, 4, 4, 64, 128])
			}

			self.e_b = {
				'h1': bias_variable([16])
			}

			self.e_bn2 = BatchNormalization([32], 'bn2')
			self.e_bn3 = BatchNormalization([64], 'bn3')
			self.e_bn4 = BatchNormalization([128], 'bn4')


	def __call__(self, z, vox_a, train):
		# get encoder. a: (batch, 2*2*2*128), 
		a = self.encoder(vox_a, train)
		# concate a and z
		z = tf.concat([a, z],1)
		# decoder
		vox_rgb = self.decoder(z)
		# apply mask
		vox_rgba = self.mask(vox_a, vox_rgb)
		return vox_rgba

	def encoder(self, vox_a, train):
		# vox_a : (batch, 32, 32, 32, 1), \in {0,1}
		a = lrelu(conv3d(vox_a, self.e_W['h1']) + self.e_b['h1'])
		a = lrelu(self.e_bn2(conv3d(a, self.e_W['h2']), train))
		a = lrelu(self.e_bn3(conv3d(a, self.e_W['h3']), train))
		a = lrelu(self.e_bn4(conv3d(a, self.e_W['h4']), train))
		a = tf.reshape(a, [-1, 2*2*2*128])
		return a

	def decoder(self, z):
		# vox_rgb : (batch, 32, 32, 32, 3),		
		shape = z.get_shape().as_list()
		#self.d_W['h1'] = weight_variable([shape[1], 2*2*2*128])

		h = tf.nn.relu(vbn(tf.matmul(z, self.d_W['h1']), 'g_vbn_1'))
		h = tf.reshape(h, [-1, 2, 2, 2, 128])
		h = tf.nn.relu(vbn(deconv3d(h, self.d_W['h2'], [shape[0], 4, 4, 4, 64]), 'g_vbn_2'))
		h = tf.nn.relu(vbn(deconv3d(h, self.d_W['h3'], [shape[0], 8, 8, 8, 32]), 'g_vbn_3'))
		h = tf.nn.relu(vbn(deconv3d(h, self.d_W['h4'], [shape[0], 16, 16, 16, 16]), 'g_vbn_4'))
		vox_rgb = tf.nn.tanh(deconv3d(h, self.d_W['h5'], [shape[0], 32, 32, 32, 3]) + self.d_b['h5'])
		return vox_rgb

	def mask(self, vox_a, vox_rgb):
		# vox_rgba : (batch, 32, 32, 32, 4),		
		vox_as = tf.concat([vox_a, vox_a, vox_a], 4)
		vox_rgb = tf.multiply(vox_rgb,vox_as)
		vox_rgba = tf.concat([vox_rgb, vox_a], 4)
		return vox_rgba


class Discriminator(object):

	def __init__(self, name="d_"):
		with tf.variable_scope(name):
			self.name = name
			self.n_kernels = 300
			self.dim_per_kernel = 50

			self.W = {
				'h1': weight_variable([4, 4, 4, 4, 16]),
				'h2': weight_variable([4, 4, 4, 16, 32]),
				'h3': weight_variable([4, 4, 4, 32, 64]),
				'h4': weight_variable([4, 4, 4, 64, 128]),
				'h5': weight_variable([2*2*2*128+self.n_kernels, 2]),
				'md': weight_variable([2*2*2*128, self.n_kernels*self.dim_per_kernel])
			}

			self.b = {
				'h1': bias_variable([16]),
				'h5': bias_variable([2]),
				'md': bias_variable([self.n_kernels])
			}

			self.bn2 = BatchNormalization([32], 'bn2')
			self.bn3 = BatchNormalization([64], 'bn3')
			self.bn4 = BatchNormalization([128], 'bn4')

	def __call__(self, x, train):
		shape = x.get_shape().as_list()		
		#noisy_x = x + tf.random_normal([shape[0], 32, 32, 32, 1])
		noisy_x = x
		h = lrelu(conv3d(noisy_x, self.W['h1']) + self.b['h1'])
		h = lrelu(self.bn2(conv3d(h, self.W['h2']), train))
		h = lrelu(self.bn3(conv3d(h, self.W['h3']), train))
		h = lrelu(self.bn4(conv3d(h, self.W['h4']), train))
		h = tf.reshape(h, [-1, 2*2*2*128])

		m = tf.matmul(h, self.W['md'])
		m = tf.reshape(m, [-1, self.n_kernels, self.dim_per_kernel])
		abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(m, 3) - tf.expand_dims(tf.transpose(m, [1, 2, 0]), 0)), 2)
		f = tf.reduce_sum(tf.exp(-abs_dif), 2) + self.b['md']

		h = tf.concat([h, f],1)
		y = tf.matmul(h, self.W['h5']) + self.b['h5']
		return y


