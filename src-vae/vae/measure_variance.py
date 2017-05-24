import tensorflow as tf
import numpy as np
import model
import dataset
import time
import os
import lapjv
from scipy.stats.distributions import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import linalg as LA

import config as myconfig

# 
def prepare_batch_dict(data_dict):
	batch_rgba = data_dict['rgba'] 
	batch_index = data_dict['index']
	batch_dict = {'rgba':batch_rgba, 'index':batch_index}
	return batch_dict

def prepare_feed_dict(batch_dict, rgb, a, mask_a, index, train, flag):
	batch_rgba = batch_dict['rgba']
	batch_rgb = batch_rgba[:,:,:,:,0:3]		# (n,64,64,64,3)
	batch_a = batch_rgba[:,:,:,:,3:4]		# (n,64,64,64,1)
	fetch_dict = {rgb:batch_rgb, a:batch_a, mask_a:batch_a, index:batch_dict['index'], train:flag}
	return fetch_dict

# a, mask_a are aligned. for test only
def prepare_shuffled_feed_dict(batch_dict, a, mask_a, index, train, flag):
	batch_rgba = np.array(batch_dict['rgba'])
	np.random.shuffle(batch_rgba)
	batch_a = batch_rgba[:,:,:,:,3:4]		# (n,64,64,64,1)
	fetch_dict = {a:batch_a, mask_a:batch_a, index:batch_dict['index'], train:flag}
	return fetch_dict

# batch_lab: (batch_size, 64,64,64,3)
def lab_denorm(batch_lab):
	de_l = batch_lab[:,:,:,:,0:1]*100
	de_ab = batch_lab[:,:,:,:,1:3]*115
	de_lab = tf.concat([de_l, de_ab], -1)
	return de_lab

def cal_covariance(batch_vox):
	N = batch_vox.shape[0]
	batch_vec = np.reshape(batch_vox, [N, -1])
	mean_vec = np.mean(batch_vec, axis=0)
	batch_vec -= mean_vec
	cov_mat = np.matmul(batch_vec, batch_vec.transpose())
	w, v = LA.eig(cov_mat)
	return np.sqrt(np.max(w))


# train_data = dataset.Dataset(myconfig.train_dataset_path,max_num=300,using_map=True)
# test_data = dataset.Dataset(myconfig.test_dataset_path,max_num=100,using_map=False)


if __name__ == '__main__':

	##################################################
	## set parameter
	##################################################
	num_train = 300
	num_test = 100
	batch_size = 100 # for 
	learning_rate = 0.1
	beta1 = 0.5
	z_size = 20
	save_interval = myconfig.save_interval
	sample_interval = myconfig.sample_interval

	test_data = dataset.Dataset(myconfig.test_dataset_path,max_num=num_test,using_map=False)

	##################################################.
	## build graph
	##################################################
	a = tf.placeholder(tf.float32, [batch_size, 64, 64, 64, 1])
	mask_a = tf.placeholder(tf.float32, [batch_size, 64, 64, 64, 1])
	rgb = tf.placeholder(tf.float32, [batch_size, 64, 64, 64, 3])
	train = tf.placeholder(tf.bool)
	indexes = tf.placeholder(tf.int32, [batch_size,])
	with tf.variable_scope('G_z'):
		all_z = tf.get_variable("prior_z", [num_train, z_size], tf.float32, tf.random_normal_initializer(stddev=1))

	# train graph
	G = model.Generator(z_size)
	rgb_, rgba_ = G.fix_shape(a, mask_a, indexes, all_z, train)
	loss_G = tf.reduce_sum(tf.abs(lab_denorm(rgb)-lab_denorm(rgb_)))/tf.reduce_sum(a)
	loss_G_list = tf.reduce_sum(tf.abs(lab_denorm(rgb)-lab_denorm(rgb_)),axis=[1,2,3,4])/tf.reduce_sum(a,axis=[1,2,3,4])	# (batch_size,)
	var_G = [v for v in tf.trainable_variables() if 'g_' in v.name]
	opt_G = tf.train.AdamOptimizer(0.001, 0.5).minimize(loss_G, var_list=var_G)
	var_z = [v for v in tf.trainable_variables() if 'prior_z' in v.name]
	opt_z = tf.train.AdamOptimizer(learning_rate, beta1).minimize(loss_G, var_list=var_z)

	print 'var_G'
	for v in var_G:
		print v
	print 'var_z'
	for v in var_z:
		print v

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		##################################################
		## Initialize variables
		##################################################
		sess.run(tf.global_variables_initializer())

		##################################################
		## sample z values according to prior distribution
		## and assign those z to each ground truth image
		## via PCA
		##################################################
		proj_images = np.load(myconfig.pca_path)[:num_train]
		print 'pca', proj_images.shape
		costM = model.pdist2(all_z, proj_images).eval()
		assignment = lapjv.lapjv(costM)[2] # get colum assignment
		buffer_batch_z = (tf.gather(all_z, assignment)).eval()
		assign_z = all_z.assign(buffer_batch_z)
		del(buffer_batch_z)
		sess.run(assign_z)
		print("Finished the pre-ordering of the z batches")

		##################################################
		## check preload model
		##################################################
		saver = tf.train.Saver(var_G)
		# saver = tf.train.Saver()
		print 'load model: ', myconfig.preload_model
		saver.restore(sess, myconfig.preload_model)
		with open(myconfig.loss_csv, 'a') as f:
			print >>f, myconfig.preload_model


		##################################################
		## error on test data
		##################################################	
		variance_list = []
		for i in xrange(num_test):
			data_dict = test_data.next_batch(1)
			data_dict['rgba'] = np.tile(data_dict['rgba'], [batch_size,1,1,1,1])
			data_dict['index'] = np.random.randint(0, num_train, size=(batch_size,))
			feed_dict = prepare_feed_dict(data_dict, rgb, a, mask_a, indexes, train,True)
			batch_rgb_ = rgb_.eval(feed_dict)
			var = cal_covariance(batch_rgb_)
			print i, var
			variance_list.append(var)
			

		# with open(myconfig.loss_csv, 'a') as f:
		# 	print >>f , 'test error %f'%(np.mean(train_error_list))
		# 	print >>f , 'test error %f'%(np.mean(loss_dict['G']))
		# 	print 'test error %f'%(np.mean(train_error_list))
		np.save('tmp/test_variances.npy', np.array(variance_list[:num_test]))










		