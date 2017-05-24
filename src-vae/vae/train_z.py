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

# train_data = dataset.Dataset(myconfig.train_dataset_path,max_num=300,using_map=True)
# test_data = dataset.Dataset(myconfig.test_dataset_path,max_num=100,using_map=False)


if __name__ == '__main__':

	##################################################
	## set parameter
	##################################################
	num_train = 300
	num_test = 100
	batch_size = 32 # for 
	learning_rate = 0.1
	beta1 = 0.5
	z_size = 20
	save_interval = myconfig.save_interval
	sample_interval = myconfig.sample_interval

	train_data = dataset.Dataset(myconfig.train_dataset_path,max_num=num_train,using_map=True)
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
		## error on train data
		##################################################	
		# for epoch in xrange(myconfig.ITER_MIN, myconfig.ITER_MAX):
		# 	loss_dict = {'G':[]}
		# 	for i in xrange(num_train/batch_size):
		# 		# read batch data and update Generator
		# 		batch_dict = prepare_batch_dict(train_data.next_batch(batch_size))
		# 		feed_dict = prepare_feed_dict(batch_dict, rgb, a, mask_a, indexes, train,True)
		# 		sess.run(opt_G, feed_dict=feed_dict)
		# 		batch_loss_G = loss_G.eval(feed_dict)
		# 		loss_dict['G'].append(batch_loss_G)
		# 	with open(myconfig.loss_csv, 'a') as f:
		# 		print >> f, '%d %f'%(epoch,np.mean(loss_dict['G']))
		# 		print epoch, np.mean(loss_dict['G'])

		# train_error_list = []
		# loss_dict = {'G':[]}
		# for i in xrange(num_train/batch_size+1):
		# 	batch_dict = prepare_batch_dict(train_data.next_batch(batch_size))
		# 	feed_dict = prepare_feed_dict(batch_dict, rgb, a, mask_a, indexes, train,True)
		# 	batch_loss_G = loss_G.eval(feed_dict)
		# 	batch_loss_G_list = loss_G_list.eval(feed_dict)
		# 	loss_dict['G'].append(batch_loss_G)
		# 	train_error_list += batch_loss_G_list.tolist()
		# with open(myconfig.loss_csv, 'a') as f:
		# 	print >>f , 'train error %f'%(np.mean(train_error_list))
		# 	print >>f , 'train error %f'%(np.mean(loss_dict['G']))
		# np.save('tmp/train_reconst_error.npy', np.array(train_error_list[:num_train]))
		# exit(0)

		##################################################
		## error on test data
		##################################################	
		train_error_list = []
		loss_dict = {'G':[]}
		for i in xrange(num_test):
			data_dict = test_data.next_batch(1)
			data_dict['rgba'] = np.tile(data_dict['rgba'], [batch_size,1,1,1,1])
			data_dict['index'] = np.random.randint(0, num_train, size=(batch_size,))
			feed_dict = prepare_feed_dict(data_dict, rgb, a, mask_a, indexes, train,True)
			batch_loss_G = loss_G.eval(feed_dict)
			batch_loss_G_list = loss_G_list.eval(feed_dict)
			loss_dict['G'].append(batch_loss_G)
			train_error_list.append(np.min(batch_loss_G_list))
			print i, batch_loss_G, np.min(batch_loss_G_list)
		with open(myconfig.loss_csv, 'a') as f:
			print >>f , 'test error %f'%(np.mean(train_error_list))
			print >>f , 'test error %f'%(np.mean(loss_dict['G']))
			print 'test error %f'%(np.mean(train_error_list))
		np.save('tmp/test_reconst_error.npy', np.array(train_error_list[:num_test]))
		exit(0)


		##################################################
		## update z on test data
		##################################################
		batch_dict = prepare_batch_dict(test_data.next_batch(batch_size))
		for epoch in xrange(myconfig.ITER_MIN, myconfig.ITER_MAX+1):
			loss_dict = {'G':[]}
			feed_dict = prepare_feed_dict(batch_dict, rgb, a, mask_a, index, train,True)
			sess.run(opt_z, feed_dict=feed_dict)

			batch_loss_G = loss_G.eval(feed_dict)
			loss_dict['G'].append(batch_loss_G)

			if epoch%50==0:
				with open(myconfig.loss_csv, 'a') as f:
					loss_G_mean = np.mean(loss_dict['G'])
					f.write("{0}, {1:.8f}\n".format(epoch, loss_G_mean))
					# print >> f, batch_z[0]
					print epoch, loss_G_mean



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
				# save ground-truth
				dataset.saveConcatVoxes2image(np.array(batch_dict['rgba'][0:8]), 
										myconfig.vox_prefix+"{0}.z.gt.jpg".format(epoch))

				# sample with z and mask_a aligned
				batch_rgba = rgba_.eval(feed_dict)
				# batch_rgba = lab_clip(batch_rgba)
				dataset.saveConcatVoxes2image(np.array(batch_rgba[0:8]), 
										myconfig.vox_prefix+"{0}.z.train.jpg".format(epoch))



