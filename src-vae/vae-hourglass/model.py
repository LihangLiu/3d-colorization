import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

def weight_variable(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.02))

def bias_variable(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

def conv3d(x, W, stride=2, padding='SAME'):
	return tf.nn.conv3d(x, W, strides=[1, stride, stride, stride, 1], padding=padding)

def deconv3d(x, W, output_shape, stride=2, padding='SAME'):
	return tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, stride, stride, stride, 1], padding=padding)

def maxpooling3d(x, ksize=2, stride=2, padding='SAME'):
	return tf.nn.max_pool3d(x, ksize=[1,ksize,ksize,ksize,1], strides=[1,stride,stride,stride,1], padding=padding)

class Generator(object):
	def __init__(self):
		pass

	def __call__(self, input_a, mask_a, z):
		rgbs = self.build(input_a, z)
		rgbs = [mask(rgb, mask_a) for rgb in rgbs]
		rgbas = [tf.concat([rgb,mask_a], -1) for rgb in rgbs]
		return rgbs, rgbas

	def fix_shape(self, a, mask_a, indexes, all_z):
		z = tf.gather(all_z, indexes)
		return self.__call__(a, mask_a, z)

	def residual(self, input, input_channels, output_channels, scope=None, reuse=None):
		with tf.variable_scope(scope, "residual", [input], reuse=reuse):
			W = weight_variable((1,1,1,input_channels,output_channels/2))
			conv = conv3d(input, W, stride=1, padding='VALID')
			W = weight_variable((3,3,3,output_channels/2,output_channels/2))
			conv1 = conv3d(conv, W, stride=1, padding='SAME')
			W = weight_variable((1,1,1,output_channels/2,output_channels))
			conv2 = conv3d(conv1, W, stride=1, padding='VALID')
			with tf.variable_scope("skip_path"):
				if input_channels == output_channels:
					skip = input
				else:
					W = weight_variable((1,1,1,input_channels,output_channels))
					skip = conv3d(input, W, stride=1, padding='VALID')
		return conv2 + skip


	def hourglass(self, input, z, num_branches, input_channels, output_channels, num_res_modules=1, scope=None, reuse=None):
		with tf.variable_scope(scope, "hourglass", [input], reuse=reuse):
			
			# Add the residual modules for the upper branch
			with tf.variable_scope("upper_branch"):
				up1 = input
				for i in range(num_res_modules):
					up1 = self.residual(up1, input_channels, input_channels)

			# Add the modules for the lower branch
			# 1. Pool -> Residuals -> Hourglass -> Residuals -> Upsample
			# 2. Pool -> Residuals -> Residuals -> Residuals -> Upsample
			with tf.variable_scope("lower_branch"):
				low1 = maxpooling3d(input, ksize=2, stride=2, padding='VALID')
				for i in range(num_res_modules):
					low1 = self.residual(low1, input_channels, input_channels)
				
				# Are we recursing? 
				if num_branches > 1:
					low2 = self.hourglass(low1, z, num_branches-1, input_channels, input_channels, num_res_modules, scope, reuse)
				else:
					# add z
					low1_b,low1_w,low1_h,low1_d,low1_c = low1.get_shape().as_list()
					z_size = z.get_shape().as_list()[1]
					W = weight_variable([z_size, low1_w*low1_h*low1_d*1])
					z = tf.matmul(z, W)		
					z = tf.reshape(z, [-1,low1_w,low1_h,low1_d,1]) 
					# concat low2 and z
					low2 = tf.concat([z,low1], -1)
					for i in range(num_res_modules):
						low2 = self.residual(low2, input_channels+1, input_channels)
			
				low3 = low2
				for i in range(num_res_modules):
					low3 = self.residual(low3, input_channels, input_channels)
				
				low3_shape = low3.get_shape().as_list()
				low3_b,low3_h,low3_w,low3_d = low3_shape[0:4]
				W = weight_variable((2,2,2,input_channels,input_channels))
				up2 = deconv3d(low3, W, [low3_b,low3_h*2,low3_w*2,low3_d*2,input_channels])	# no nearest neighbor for 3d

			return up1 + up2

	def build(self, input, z, num_parts=3, num_features=64, num_stacks=2, num_res_modules=1, reuse=None, scope='g_'):

		with tf.variable_scope(scope, 'StackedHourGlassNetwork', [input], reuse=reuse):
			
			# Initial processing of the image
			r3 = self.residual(input, 1, num_features)

			intermediate_features = r3

			rgbs = []
			for i in range(num_stacks):
				print i
				# Build the hourglass
				hg = self.hourglass(intermediate_features, z, num_branches=4, input_channels=num_features, output_channels=num_features)
				
				# Residual layers at the output resolution
				ll = hg
				for j in range(num_res_modules):
					ll = self.residual(ll, num_features, num_features)
				
				# Linear layers to produce the first set of predictions
				W = weight_variable((1,1,1,num_features,num_features))
				ll = conv3d(ll, W, stride=1, padding='VALID')

				# Predicted rgbs
				W = weight_variable((1,1,1,num_features,num_parts))
				rgb = conv3d(ll, W, stride=1, padding='VALID')
				rgbs.append(rgb)

				# Add the predictions back
				if i < num_stacks - 1:
					W = weight_variable((1,1,1,num_features,num_features))
					ll_ = conv3d(ll, W, stride=1, padding='VALID')
					W = weight_variable((1,1,1,num_parts,num_features))
					heatmap_ = conv3d(rgb, W, stride=1, padding='VALID')
					intermediate_features = intermediate_features + ll_ + heatmap_
			
		return rgbs



def mask(rgb, a):
### rgb \in (-1,1), (batch, 32, 32, 32, 3)
### a \in {-1,1},		 (batch, 32, 32, 32, 1)
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


if __name__ == '__main__':
	input_a = tf.placeholder(tf.float32, [32, 64, 64, 64, 1])
	mask_a = tf.placeholder(tf.float32, [32, 64, 64, 64, 1])
	z = tf.placeholder(tf.float32, [32, 20])
	G = Generator()
	rgbs,rgbas = G(input_a, mask_a, z)
	print len(rgbs), len(rgbas)
	print rgbs[0].get_shape().as_list()
	print rgbas[0].get_shape().as_list()
	count = 0
	for v in tf.trainable_variables():
		count += np.sum(v.get_shape().as_list())
	print count
	print len(tf.trainable_variables())






