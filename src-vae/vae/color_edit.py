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


if __name__ == '__main__':

	##################################################
	## set parameter
	##################################################
	num_train = 64
	num_test = 100
	batch_size = 32 # for 
	learning_rate = 0.01
	beta1 = 0.5
	z_size = 20
	save_interval = myconfig.save_interval
	sample_interval = myconfig.sample_interval

	train_data = dataset.Dataset(myconfig.train_dataset_path,max_num=num_train,using_map=True)

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

	z = tf.placeholder(tf.float32, [batch_size, z_size])

	# train graph
	G = model.Generator(z_size)
	rgb_, rgba_ = G.fix_shape(a, mask_a, indexes, all_z, train)
	loss_G = tf.reduce_sum(tf.abs(lab_denorm(rgb)-lab_denorm(rgb_)))/tf.reduce_sum(a)
	loss_G_list = tf.reduce_sum(tf.abs(lab_denorm(rgb)-lab_denorm(rgb_)),axis=[1,2,3,4])/tf.reduce_sum(a,axis=[1,2,3,4])	# (batch_size,)
	var_G = [v for v in tf.trainable_variables() if 'g_' in v.name]
	var_z = [v for v in tf.trainable_variables() if 'prior_z' in v.name]
	opt_G = tf.train.AdamOptimizer(0.001, 0.5).minimize(loss_G, var_list=var_G)
	opt_z = tf.train.AdamOptimizer(learning_rate, beta1).minimize(loss_G, var_list=var_z)

	_, rgba_randz_ = G(a, mask_a, z, train)

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
		## train
		##################################################	
		for epoch in xrange(myconfig.ITER_MIN, myconfig.ITER_MAX):
			loss_dict = {'G':[]}
			for i in xrange(num_train/batch_size):
				# read batch data and update Generator
				batch_dict = prepare_batch_dict(train_data.next_batch(batch_size))
				feed_dict = prepare_feed_dict(batch_dict, rgb, a, mask_a, indexes, train,True)
				sess.run(opt_G, feed_dict=feed_dict)
				batch_loss_G = loss_G.eval(feed_dict)
				loss_dict['G'].append(batch_loss_G)
			# with open(myconfig.loss_csv, 'a') as f:
			# 	print >> f, '%d %f'%(epoch,np.mean(loss_dict['G']))
			print epoch, np.mean(loss_dict['G'])


		##################################################
		## color editing by interpolating z
		##################################################
		# source
		src_batch_dict = prepare_batch_dict(train_data.next_batch(batch_size))
		src_feed_dict = prepare_feed_dict(src_batch_dict, rgb, a, mask_a, indexes, train,True)
		src_batch_rgba = rgba_.eval(src_feed_dict)
		# target
		tar_batch_dict = prepare_batch_dict(train_data.next_batch(batch_size))
		tar_feed_dict = prepare_feed_dict(tar_batch_dict, rgb, a, mask_a, indexes, train,True)
		tar_batch_rgba = rgba_.eval(tar_feed_dict)
		# interpolate
		src_batch_z = tf.gather(all_z, indexes).eval(src_feed_dict)
		tar_batch_z = tf.gather(all_z, indexes).eval(tar_feed_dict)
		int_batch_z = np.concatenate([src_batch_z[:,:z_size/2], tar_batch_z[:,z_size/2:]], -1)
		tar_feed_dict[z] = int_batch_z
		# tar_feed_dict[z] = (src_batch_z + tar_batch_z)/2
		int_batch_rgba = rgba_randz_.eval(tar_feed_dict)

		##################################################
		## draw samples on train and test dataset
		## 
		##################################################

		def lab_clip(batch_laba):
			batch_laba = np.array(batch_laba)
			batch_laba[:,:,:,:,0] = np.clip(batch_laba[:,:,:,:,0],0,1)
			batch_laba[:,:,:,:,1] = np.clip(batch_laba[:,:,:,:,1],-1,1)
			batch_laba[:,:,:,:,2] = np.clip(batch_laba[:,:,:,:,2],-1,1)
			return batch_laba

		dataset.saveConcatVoxes2image(np.array(src_batch_rgba[:12]), "tmp/src_z.jpg")
		dataset.saveConcatVoxes2image(np.array(tar_batch_rgba[:12]), "tmp/tar_z.jpg")
		dataset.saveConcatVoxes2image(np.array(int_batch_rgba[:12]), "tmp/int_z.jpg")




