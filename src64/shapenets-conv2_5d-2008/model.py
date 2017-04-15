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

# filter: (D,H,2,C_in,C_out) | (D,2,W,C_in,C_out) | (2,H,W,C_in,C_out)
# 	among the 2: (value v, depth d). 
# -> filter_3d: (D,H,W,C_in,C_out)

ALPHA_NAME = "alpha"    # not clip by d

def flt2_5d_to_3d(filter, axis, depth):
	shape_3d = filter.get_shape().as_list()
	#
	if axis not in [0,1,2]:
		print "axis must be 0 or 1 or 2, instead of ", axis
		exit(0)
	if shape_3d[axis] != 2:
		print "filter size doesn't match", shape_3d, axis
		exit(0)

	# list of depth of filiter_3d, [-D/2, D/2]
	D = 0.02		# factor
	#with tf.variable_scope(ALPHA_NAME):
	alpha = 0.1*bias_variable([])	# init to be 0.01
	di_s = (np.arange(0.0,depth,1.0)/(depth-1)-0.5)*D #(-0.01,0.01)

	v,d = tf.split(filter, num_or_size_splits=2, axis=axis)
	# filter_3d = [v*D/(D+alpha*tf.abs(d-di_s[i])) for i in range(depth)]
	filter_3d = [v*tf.exp(-alpha*tf.square((depth-1)*(d-di_s[i])/D))
												for i in range(depth)]
	filter_3d = tf.concat(filter_3d, axis)
	print di_s
	print shape_3d
	print v.get_shape()
	print d.get_shape()
	print filter_3d.get_shape()
	return filter_3d

def conv2_5d(x, filter, axis=2, depth=4, stride=2):
	filter_3d = flt2_5d_to_3d(filter, axis, depth)

	res = conv3d(x, filter_3d, stride=stride)
	print res.get_shape()
	return res

# recommended shape: (4,4,4,c_in,c_out)
# c_out : 4*k
def weights_2_5_d(shape):
	if len(shape) != 5:
		print "wrong dimensions, should be 5"
		exit(0)

	filter_3d = []
	c_out = shape[-1]
	# conv2_5d
	for axis in [0,1,2]:		# projected axises
		new_shape = shape[:]	# must be deep copy here
		new_shape[axis] = 2
		new_shape[-1] = c_out/4
		filter2_5d = weight_variable(new_shape)
		filter_3d.append(flt2_5d_to_3d(filter2_5d, axis, shape[axis]))
	# conv3d
	new_shape = shape[:]    # must be deep copy here
	new_shape[-1] = c_out/4
	filter_3d.append(weight_variable(new_shape))
	filter_3d = tf.concat(filter_3d, -1)
	print filter_3d.get_shape()
	return filter_3d



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



class Discriminator(object):

	def __init__(self, ndf=16, name="d_"):
		with tf.variable_scope(name):
			self.name = name
			self.ndf = ndf

			self.W = {
				'h1': weights_2_5_d([4, 4, 4, 4, ndf]),
				'h2': weights_2_5_d([4, 4, 4, ndf, ndf*2]),
				'h3': weights_2_5_d([4, 4, 4, ndf*2, ndf*4]),
				'h4': weights_2_5_d([4, 4, 4, ndf*4, ndf*8]),
				'h5': weights_2_5_d([4, 4, 4, ndf*8, ndf*16]),
				'h6': weights_2_5_d([4, 4, 4, ndf*16, ndf*32]),
				'fc1': weight_variable([1*1*1*ndf*32, 1024]),
				'fc2': weight_variable([1024, 55])
			}

			self.b = {
				'h1': bias_variable([ndf]),
				'fc1': bias_variable([1024]),
				'fc2': bias_variable([55])
			}

			self.bn2 = BatchNormalization([ndf*2], 'bn2')
			self.bn3 = BatchNormalization([ndf*4], 'bn3')
			self.bn4 = BatchNormalization([ndf*8], 'bn4')
			self.bn5 = BatchNormalization([ndf*16], 'bn5')
			self.bn6 = BatchNormalization([ndf*32], 'bn6')

	def __call__(self, x, train):
		shape = x.get_shape().as_list()		#(n,64,64,64,4)
		#noisy_x = x + tf.random_normal(shape,mean=0.0,stddev=1)
		noisy_x = x
		
		h1 = lrelu(conv3d(noisy_x, self.W['h1']) + self.b['h1'])	#(n,32,32,32,f)
		h2 = lrelu(self.bn2(conv3d(h1, self.W['h2']), train))		#(n,16,16,16,f*2)
		h3 = lrelu(self.bn3(conv3d(h2, self.W['h3']), train))		#(n,8,8,8,f*4)
		h4 = lrelu(self.bn4(conv3d(h3, self.W['h4']), train))		#(n,4,4,4,f*8)
		h5 = lrelu(self.bn5(conv3d(h4, self.W['h5']), train))		#(n,2,2,2,f*16)
		h6 = lrelu(self.bn6(conv3d(h5, self.W['h6']), train))		#(n,1,1,1,f*32)
		h = tf.reshape(h6, [-1, 1*1*1*self.ndf*32])

		fc1 = tf.matmul(h, self.W['fc1']) + self.b['fc1']	#(n,1024)
		y = tf.matmul(fc1, self.W['fc2']) + self.b['fc2']		#(n,55)
		return y












