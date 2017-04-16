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
	batch_rgb = batch_rgba[:,:,:,:,0:3]		# (n,64,64,64,3)
	batch_a = batch_rgba[:,:,:,:,3:4]		# (n,64,64,64,1)
	batch_index = data_dict['index']
	batch_dict = {'rgba':batch_rgba, 'rgb':batch_rgb, 'a':batch_a, 'index':batch_index}
	return batch_dict

def prepare_feed_dict(batch_dict, rgb, a, index, train, flag):
	fetch_dict = {rgb:batch_dict['rgb'], a:batch_dict['a'],index:batch_dict['index'], train:flag}
	return fetch_dict

def prepare_fix_shape_feed_dict(batch_dict, rgb, a, index, train, flag):
	batch_rgb = np.array(batch_dict['rgb'])
	batch_index = np.array(batch_dict['index'])
	# shuffle index and rgb
	c = list(zip(batch_rgb, batch_index))
	np.random.shuffle(c)
	batch_rgb, batch_index = zip(*c)
	fetch_dict = {rgb:np.array(batch_rgb), a:batch_dict['a'],index:np.array(batch_index), train:flag}
	return fetch_dict

train_data = dataset.Dataset(myconfig.train_dataset_path)
test_data = dataset.Dataset(myconfig.test_dataset_path)

if __name__ == '__main__':

	##################################################
	## set parameter
	##################################################
	batch_size = 32 # for 
	num_train = train_data.num_examples
	learning_rate = 0.0001*100
	beta1 = 0.5
	z_size = 20
	save_interval = 100
	sample_interval = 10
	total_batch = num_train / batch_size


	##################################################.
	## build graph
	##################################################
	a = tf.placeholder(tf.float32, [batch_size, 64, 64, 64, 1])
	rgb = tf.placeholder(tf.float32, [batch_size, 64, 64, 64, 3])
	train = tf.placeholder(tf.bool)
	indexes = tf.placeholder(tf.int32, [batch_size,])
	with tf.variable_scope('G_z'):
		all_z = tf.get_variable("prior_z", [num_train, z_size], tf.float32, tf.random_normal_initializer(stddev=1))

	z = tf.placeholder(tf.float32, [batch_size, z_size])

	# train graph
	G = model.Generator(z_size)
	rgb_, rgba_ = G.train(a, indexes, all_z, train)
	loss_G = tf.reduce_mean(tf.abs(rgb-rgb_))
	var_G = [v for v in tf.trainable_variables() if 'g_' in v.name]
	opt_G = tf.train.AdamOptimizer(learning_rate, beta1).minimize(loss_G, var_list=var_G)

	# sample graph
	sample_rgb_, sample_rgba_ = G.sample(a, z, train)

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
		proj_images = np.load(myconfig.pca_path)
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
		saver = tf.train.Saver()
		if myconfig.preload_model:
			print 'load model: ', myconfig.preload_model
			saver.restore(sess, myconfig.preload_model)

		##################################################
		## train
		##################################################
		for epoch in xrange(myconfig.ITER_MIN, myconfig.ITER_MAX):
			loss_dict = {'G':[], 'G_fix':[]}
			for i in xrange(total_batch):
				# read batch data
				batch_dict = prepare_batch_dict(train_data.next_batch(batch_size))
				# update Generator
				feed_dict = prepare_feed_dict(batch_dict, rgb, a, indexes, train,True)
				sess.run(opt_G, feed_dict=feed_dict)

				# update fix shape regularization
				# if epoch%3 == -1:
				feed_dict = prepare_fix_shape_feed_dict(batch_dict, rgb, a, indexes, train,True)
				sess.run(opt_G, feed_dict=feed_dict)

				with open(myconfig.log_txt, 'a') as f:
					feed_dict = prepare_feed_dict(batch_dict, rgb, a, indexes, train,False)
					batch_loss_G = loss_G.eval(feed_dict)
					feed_dict = prepare_fix_shape_feed_dict(batch_dict, rgb, a, indexes, train,False)
					batch_loss_G_fix = loss_G.eval(feed_dict)
					loss_dict['G'].append(batch_loss_G)
					loss_dict['G_fix'].append(batch_loss_G_fix)
					msg = "{0}, {1}, {2:.8f}, {3:.8f}".format(epoch, i, batch_loss_G, batch_loss_G_fix)
					print >> f, msg
					print msg

			with open(myconfig.loss_csv, 'a') as f:
				loss_G_mean = np.mean(loss_dict['G'])
				loss_G_fix_mean = np.mean(loss_dict['G_fix'])
				f.write("{0}, {1:.8f}, {2:.8f}\n".format(epoch, loss_G_mean, loss_G_fix_mean))


			##################################################
			## draw samples on train and test dataset
			## 
			##################################################
			def saveConcatVoxes2image(voxes, imname):
				sub_names = []
				for i,vox in enumerate(voxes): 
					# print i,vox
					sub_name = "tmp/tmp-%d.jpg"%(i)
					dataset.vox2image(vox, sub_name)
					sub_names.append(sub_name)
				dataset.concatenateImages(sub_names, imname)
				print imname
				for name in sub_names:
					os.remove(name)

			if epoch % sample_interval == 0:
				batch_dict = prepare_batch_dict(train_data.next_batch(batch_size))

				# save ground-truth
				saveConcatVoxes2image(dataset.transformBack(np.array(batch_dict['rgba'][0:12])), 
										myconfig.vox_prefix+"{0}.gt.jpg".format(epoch))

				# sample train z
				feed_dict = prepare_feed_dict(batch_dict, rgb, a, indexes, train,True)
				batch_rgba = rgba_.eval(feed_dict)
				saveConcatVoxes2image(dataset.transformBack(np.array(batch_rgba[0:12])), 
										myconfig.vox_prefix+"{0}.train.jpg".format(epoch))

				# sample random z
				random_z = np.random.normal(0, 1, [batch_size, z_size]).astype(np.float32)
				for n in range(5):	# iterate on 5 rgbas
					c_a = np.array(batch_dict['a'][n:n+1])
					rep_a = np.tile(c_a, (batch_size, 1, 1, 1, 1))
					batch_rgba = sample_rgba_.eval({a:rep_a, z:random_z, train:False})
					saveConcatVoxes2image(dataset.transformBack(np.array(batch_rgba[0:12])), 
											myconfig.vox_prefix+"{0}-{1}.test.jpg".format(epoch, n))


			##################################################
			## save network parameter
			##################################################
			if epoch % save_interval == 0:
				saver.save(sess, myconfig.param_prefix+"{0}.ckpt".format(epoch))
			


