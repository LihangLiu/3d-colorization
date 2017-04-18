import tensorflow as tf
import numpy as np


def weight_variable(name, shape):
	return tf.get_variable(name, shape, tf.float32, tf.random_normal_initializer(stddev = 0.02))

def bias_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1)),

def addRandomNormal(x, stddev=1.0):
	return x + tf.random_normal(x.get_shape().as_list(),stddev=stddev)

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


### 
### 
class Generator(object):

	def __init__(self, z_size=5, ngf=9, name="g_"):
		with tf.variable_scope(name):
			self.name = name
			self.ngf = ngf

			self.W = {
				'hz': weight_variable('hz', [z_size, 64*64*64*1]),

				'h1': weight_variable('h1', [4, 4, 4, 2, ngf]),
				'h2': weight_variable('h2', [4, 4, 4, ngf, ngf*2]),
				'h3': weight_variable('h3', [4, 4, 4, ngf*2, ngf*4]),
				'h4': weight_variable('h4', [4, 4, 4, ngf*4, ngf*8]),
				'h5': weight_variable('h5', [4, 4, 4, ngf*8, ngf*16]),
				'h6': weight_variable('h6', [4, 4, 4, ngf*16, ngf*32]),

				'dh1': weight_variable('dh1', [4, 4, 4, ngf*16, ngf*32]),
				'dh2': weight_variable('dh2', [4, 4, 4, ngf*8, ngf*16]),
				'dh3': weight_variable('dh3', [4, 4, 4, ngf*4, ngf*8]),
				'dh4': weight_variable('dh4', [4, 4, 4, ngf*2, ngf*4]),
				'dh5': weight_variable('dh5', [4, 4, 4, ngf, ngf*2]),
				'dh6': weight_variable('dh6', [4, 4, 4, 3, ngf])
			}

			self.b = {
				'h1': bias_variable('bias-h1', [ngf]),
				'dh6': bias_variable('bias-dh6', [3]),
			}

			self.bn2 = batch_norm(name = 'bn2')
			self.bn3 = batch_norm(name = 'bn3')
			self.bn4 = batch_norm(name = 'bn4')
			self.bn5 = batch_norm(name = 'bn5')
			self.bn6 = batch_norm(name = 'bn6')

			self.dbn1 = batch_norm(name = 'dbn1')
			self.dbn2 = batch_norm(name = 'dbn2')
			self.dbn3 = batch_norm(name = 'dbn3')
			self.dbn4 = batch_norm(name = 'dbn4')
			self.dbn5 = batch_norm(name = 'dbn5')

	def __call__(self, a, z, train):
		shape = a.get_shape().as_list()		# (n,64,64,64,1)
		N = shape[0]
		ngf = self.ngf

		# add noise
		z = tf.matmul(z, self.W['hz'])		# z:(n,64*64*64)
		z = tf.reshape(z, [-1,64,64,64,1]) 	# z:(n,64,64,64,1)
		h = tf.concat([a,z], -1)			# (n,64,64,64,2)

		# conv
		h1 = lrelu(conv3d(h,self.W['h1']) + self.b['h1'])	# (n,32,32,32,f)
		h2 = lrelu(self.bn2(conv3d(h1,self.W['h2']), train)) # (n,16,16,16,f*2)
		h3 = lrelu(self.bn3(conv3d(h2,self.W['h3']), train)) # (n,8,8,8,f*4)
		h4 = lrelu(self.bn4(conv3d(h3,self.W['h4']), train)) # (n,4,4,4,f*8)
		h5 = lrelu(self.bn5(conv3d(h4,self.W['h5']), train)) # (n,2,2,2,f*16)
		h6 = lrelu(self.bn6(conv3d(h5,self.W['h6']), train)) # (n,1,1,1,f*32)

		# deconv
		dh1 = tf.nn.relu(self.dbn1(deconv3d(h6, self.W['dh1'], [N,2,2,2,ngf*16]), train)) #(n,2,2,2,f*16)
		dh2 = tf.nn.relu(self.dbn2(deconv3d(dh1, self.W['dh2'], [N,4,4,4,ngf*8]), train)) #(n,4,4,4,f*8)
		dh3 = tf.nn.relu(self.dbn3(deconv3d(dh2, self.W['dh3'], [N,8,8,8,ngf*4]), train)) #(n,8,8,8,f*4)
		dh4 = tf.nn.relu(self.dbn4(deconv3d(dh3, self.W['dh4'], [N,16,16,16,ngf*2]), train)) #(n,16,16,16,f*2)
		dh5 = tf.nn.relu(self.dbn5(deconv3d(dh4, self.W['dh5'], [N,32,32,32,ngf]), train)) #(n,32,32,32,f)
		rgb = tf.nn.tanh(deconv3d(dh5, self.W['dh6'], [N,64,64,64,3]) + self.b['dh6']) 	#(n,64,64,64,3)

		# mask
		rgba = self.mask(rgb,a)

		return rgba

	def mask(self, rgb, a):
		### rgb \in (-1,1), (batch, 64, 64, 64, 3)
		### a \in {-1,1},	(batch, 64, 64, 64, 1)
		# (-1,1) -> (0,1)
		rgb = rgb*0.5 + 0.5
		a = a*0.5 + 0.5
		# mask
		rep_a = tf.concat([a,a,a], -1)
		rgb = tf.multiply(rgb,rep_a)
		rgba = tf.concat([rgb,a], -1)
		# (0,1) -> (-1,1)
		rgba = (rgba-0.5)*2

		return rgba


class Discriminator(object):

	def __init__(self, z_size=5, ngf=9, name="d_"):
		with tf.variable_scope(name):
			self.name = name
			self.ngf = ngf

			self.W = {
				'h1': weight_variable('h1', [4, 4, 4, 4, ngf]),
				'h2': weight_variable('h2', [4, 4, 4, ngf, ngf*2]),
				'h3': weight_variable('h3', [4, 4, 4, ngf*2, ngf*4]),
				'h4': weight_variable('h4', [4, 4, 4, ngf*4, ngf*8]),
				'h5': weight_variable('h5', [4, 4, 4, ngf*8, ngf*16]),
				'h6': weight_variable('h6', [4, 4, 4, ngf*16, ngf*32]),

				'dh1': weight_variable('dh1', [4, 4, 4, ngf*16, ngf*32]),
				'dh2': weight_variable('dh2', [4, 4, 4, ngf*8, ngf*16]),
				'dh3': weight_variable('dh3', [4, 4, 4, ngf*4, ngf*8]),
				'dh4': weight_variable('dh4', [4, 4, 4, ngf*2, ngf*4]),
				'dh5': weight_variable('dh5', [4, 4, 4, ngf, ngf*2]),
				'dh6': weight_variable('dh6', [4, 4, 4, 4, ngf])
			}

			self.b = {
				'h1': bias_variable('bias-h1', [ngf]),
				'dh6': bias_variable('bias-dh6', [4]),
			}

			self.bn2 = batch_norm(name = 'd-bn2')
			self.bn3 = batch_norm(name = 'd-bn3')
			self.bn4 = batch_norm(name = 'd-bn4')
			self.bn5 = batch_norm(name = 'd-bn5')
			self.bn6 = batch_norm(name = 'd-bn6')

			self.dbn1 = batch_norm(name = 'd-dbn1')
			self.dbn2 = batch_norm(name = 'd-dbn2')
			self.dbn3 = batch_norm(name = 'd-dbn3')
			self.dbn4 = batch_norm(name = 'd-dbn4')
			self.dbn5 = batch_norm(name = 'd-dbn5')

	def __call__(self, rgba, train):
		shape = rgba.get_shape().as_list()		# (n,64,64,64,1)
		N = shape[0]
		ngf = self.ngf

		# conv
		h1 = lrelu(conv3d(rgba,self.W['h1']) + self.b['h1'])	# (n,32,32,32,f)
		h2 = lrelu(self.bn2(conv3d(h1,self.W['h2']), train)) # (n,16,16,16,f*2)
		h3 = lrelu(self.bn3(conv3d(h2,self.W['h3']), train)) # (n,8,8,8,f*4)
		h4 = lrelu(self.bn4(conv3d(h3,self.W['h4']), train)) # (n,4,4,4,f*8)
		h5 = lrelu(self.bn5(conv3d(h4,self.W['h5']), train)) # (n,2,2,2,f*16)
		h6 = lrelu(self.bn6(conv3d(h5,self.W['h6']), train)) # (n,1,1,1,f*32)

		# deconv
		dh1 = tf.nn.relu(self.dbn1(deconv3d(h6, self.W['dh1'], [N,2,2,2,ngf*16]), train)) #(n,2,2,2,f*16)
		dh2 = tf.nn.relu(self.dbn2(deconv3d(dh1, self.W['dh2'], [N,4,4,4,ngf*8]), train)) #(n,4,4,4,f*8)
		dh3 = tf.nn.relu(self.dbn3(deconv3d(dh2, self.W['dh3'], [N,8,8,8,ngf*4]), train)) #(n,8,8,8,f*4)
		dh4 = tf.nn.relu(self.dbn4(deconv3d(dh3, self.W['dh4'], [N,16,16,16,ngf*2]), train)) #(n,16,16,16,f*2)
		dh5 = tf.nn.relu(self.dbn5(deconv3d(dh4, self.W['dh5'], [N,32,32,32,ngf]), train)) #(n,32,32,32,f)
		rgba_out = tf.nn.tanh(deconv3d(dh5, self.W['dh6'], [N,64,64,64,4]) + self.b['dh6']) 	#(n,64,64,64,4)

		return rgba_out


def loss(D_real_in, D_real_out, D_gen_in, D_gen_out, k_t, gamma=0.75):
	'''
	The Bounrdary Equibilibrium GAN uses an approximation of the
	Wasserstein Loss between the disitributions of pixel-wise
	autoencoder loss based on the discriminator performance on
	real vs. generated data.
	This simplifies to reducing the L1 norm of the autoencoder loss:
	making the discriminator objective to perform well on real images
	and poorly on generated images; with the generator objective
	to create samples which the discriminator will perform well upon.
	args:
		D_real_in:  input to discriminator with real sample.
		D_real_out: output from discriminator with real sample.
		D_gen_in: input to discriminator with generated sample.
		D_gen_out: output from discriminator with generated sample.
		k_t: weighting parameter which constantly updates during training
		gamma: diversity ratio, used to control model equibilibrium.
	returns:
		D_loss:  discriminator loss to minimise.
		G_loss:  generator loss to minimise.
		k_tp:	value of k_t for next train step.
		convergence_measure: measure of model convergence.
	'''
	def pixel_autoencoder_loss(out, inp):
		'''
		The autoencoder loss used is the L1 norm (note that this
		is based on the pixel-wise distribution of losses
		that the authors assert approximates the Normal distribution)
		args:
			out:  discriminator output
			inp:  discriminator input
		returns:
			L1 norm of pixel-wise loss
		'''
		eta = 1  # paper uses L1 norm
		diff = tf.abs(out - inp)
		if eta == 1:
			# return tf.reduce_sum(diff)
			return tf.reduce_mean(diff)
		else:
			# return tf.reduce_sum(tf.pow(diff, eta))
			return tf.reduce_mean(tf.pow(diff, eta))

	mu_real = pixel_autoencoder_loss(D_real_out, D_real_in)
	mu_gen = pixel_autoencoder_loss(D_gen_out, D_gen_in)
	D_loss = mu_real - k_t * mu_gen
	G_loss = mu_gen
	lam = 0.001  # 'learning rate' for k. Berthelot et al. use 0.001
	k_tp = k_t + lam * (gamma * mu_real - mu_gen)
	convergence_measure = mu_real + np.abs(gamma * mu_real - mu_gen)
	return D_loss, G_loss, k_tp, convergence_measure








