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

train_data = dataset.Dataset(myconfig.train_dataset_path,max_num=300,using_map=True)
# test_data = dataset.Dataset(myconfig.test_dataset_path,max_num=100,using_map=False)


if __name__ == '__main__':

	##################################################
	## set parameter
	##################################################
	batch_size = 32 # for 
	num_train = train_data.num_examples
	learning_rate = 0.001
	beta1 = 0.5
	z_size = 20
	save_interval = myconfig.save_interval
	sample_interval = myconfig.sample_interval
	total_batch = num_train / batch_size


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
	loss_G = tf.reduce_mean(tf.abs(rgb-rgb_))
	var_G = [v for v in tf.trainable_variables() if 'g_' in v.name]
	opt_G = tf.train.AdamOptimizer(learning_rate, beta1).minimize(loss_G, var_list=var_G)

	# sample graph
	# sample_rgb_, sample_rgba_ = G.sample(a, z, train)

	print 'var_G'
	for v in var_G:
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
		saver = tf.train.Saver(max_to_keep=0)
		if myconfig.preload_model:
			print 'load model: ', myconfig.preload_model
			saver.restore(sess, myconfig.preload_model)

		##################################################
		## train
		##################################################
		for epoch in xrange(myconfig.ITER_MIN, myconfig.ITER_MAX):
			loss_dict = {'G':[]}
			for i in xrange(total_batch):
				# read batch data and update Generator
				batch_dict = prepare_batch_dict(train_data.next_batch(batch_size))
				feed_dict = prepare_feed_dict(batch_dict, rgb, a, mask_a, indexes, train,True)
				sess.run(opt_G, feed_dict=feed_dict)

				with open(myconfig.log_txt, 'a') as f:
					batch_dict = prepare_batch_dict(train_data.next_batch(batch_size))
					feed_dict = prepare_feed_dict(batch_dict, rgb, a, mask_a, indexes, train,False)
					batch_loss_G = loss_G.eval(feed_dict)
					loss_dict['G'].append(batch_loss_G)
					msg = "{0}, {1}, {2:.8f}".format(epoch, i, batch_loss_G)
					print >> f, msg
					print msg

			with open(myconfig.loss_csv, 'a') as f:
				loss_G_mean = np.mean(loss_dict['G'])
				f.write("{0}, {1:.8f}\n".format(epoch, loss_G_mean))


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

			if epoch % sample_interval == 0:
				batch_dict = prepare_batch_dict(train_data.next_batch(batch_size))

				# save ground-truth
				dataset.saveConcatVoxes2image(np.array(batch_dict['rgba'][0:8]), 
										myconfig.vox_prefix+"{0}.gt.jpg".format(epoch))

				# sample with z and mask_a aligned
				feed_dict = prepare_feed_dict(batch_dict, rgb, a, mask_a, indexes, train,False)
				batch_rgba = rgba_.eval(feed_dict)
				# batch_rgba = lab_clip(batch_rgba)
				dataset.saveConcatVoxes2image(np.array(batch_rgba[0:8]), 
										myconfig.vox_prefix+"{0}.train.jpg".format(epoch))

				# sample with a and mask_a being identical
				feed_dict = prepare_shuffled_feed_dict(batch_dict, a, mask_a, indexes, train,False)
				batch_rgba = rgba_.eval(feed_dict)
				# batch_rgba = lab_clip(batch_rgba)
				dataset.saveConcatVoxes2image(np.array(batch_rgba[0:8]), 
										myconfig.vox_prefix+"{0}.test.jpg".format(epoch))



			##################################################
			## save network parameter
			##################################################
			if epoch % save_interval == 0:
				saver.save(sess, myconfig.param_prefix+"{0}.ckpt".format(epoch))
			


