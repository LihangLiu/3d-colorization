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


train_data = dataset.Dataset(myconfig.train_dataset_path)
test_data = dataset.Dataset(myconfig.test_dataset_path)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:


	##################################################
	## set parameter
	##################################################
	batch_size = 32 # for 
	num_train = train_data.num_examples
	learning_rate = 0.0001*200
	beta1 = 0.5
	z_size = 20
	save_interval = 100
	sample_interval = myconfig.save_interval
	total_batch = num_train / batch_size


	##################################################.
	## build graph
	##################################################
	a = tf.placeholder(tf.float32, [batch_size, 32, 32, 32, 1])
	rgb = tf.placeholder(tf.float32, [batch_size, 32, 32, 32, 3])
	train = tf.placeholder(tf.bool)
	indexes = tf.placeholder(tf.int32, [batch_size,])
	with tf.variable_scope('G_z'):
		all_z = tf.get_variable("prior_z", [num_train, z_size], tf.float32, tf.random_normal_initializer(stddev=1))

	z = tf.placeholder(tf.float32, [batch_size, z_size])

	# generator
	G = model.Generator(z_size)
	rgb_ = G(a, indexes, all_z, train)

	loss_G = tf.reduce_mean(tf.abs(rgb-rgb_))
	var_G = [v for v in tf.trainable_variables() if 'g_' in v.name]
	opt_G = tf.train.AdamOptimizer(learning_rate, beta1).minimize(loss_G, var_list=var_G)
	
	# fix shape regularization
	rgb_fix_shape = G.fix_shape(a, indexes, all_z, train)

	loss_G_fix_shape = tf.reduce_mean(tf.abs(tf.tile(rgb,[batch_size,1,1,1,1])-rgb_fix_shape))
	opt_G_fix_shape = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_G_fix_shape, var_list=var_G, name='sgd2')

	# test
	sampler = G.sampler(a, z, train)

	print 'var_G'
	for v in var_G:
		print v

	##################################################
	## Initialize variables
	##################################################
	sess.run(tf.global_variables_initializer())

	##################################################
	## sample z values according to prior distribution
	## and assign those z to each ground truth image
	## via PCA
	##################################################
	proj_images = np.load(myconfig.pca_path)[:1000]
	costM = model.pdist2(all_z, proj_images).eval()
	assignment = lapjv.lapjv(costM)[2] # get colum assignment
	buffer_batch_z = (tf.gather(all_z, assignment)).eval()
	assign_z = all_z.assign(buffer_batch_z)
	del(buffer_batch_z)
	sess.run(assign_z)
	print("Finished the pre-ordering of the z batches")

	saver = tf.train.Saver()
	if myconfig.preload_model:
		print 'load model: ', myconfig.preload_model
		saver.restore(sess, myconfig.preload_model)
	for epoch in xrange(myconfig.ITER_MIN, myconfig.ITER_MAX):
		loss_dict = {'G':[]}
		for i in xrange(total_batch):
			# update Generator
			# read batch data
			indexes_, real_images = train_data.next_batch(batch_size)
			real_images = real_images['rgba'].astype(np.float32)
			
			sess.run(opt_G, feed_dict={rgb:real_images[:,:,:,:,:3], a:real_images[:,:,:,:,3:4], indexes:indexes_, train:True})

			# update fix shape regularization
			if epoch%3 == -1:
				sess.run(opt_G_fix_shape, feed_dict={rgb:real_images[:,:,:,:,:3], a: real_images[:,:,:,:,3:4], indexes: indexes_, train:True})

			with open(myconfig.log_txt, 'a') as f:
				batch_loss_G = loss_G.eval({rgb: real_images[:,:,:,:,:3], a: real_images[:,:,:,:,3:4], indexes: indexes_, train:False})
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
			indexes_, real_images = train_data.next_batch(batch_size)
			real_images = real_images['rgba'].astype(np.float32)
			real_rgbs = np.array(real_images[:,:,:,:,:3])
			real_as = np.array(real_images[:,:,:,:,3:4])
			# print real_as
			print indexes_
			random_z = np.random.normal(0, 1, [batch_size, z_size]).astype(np.float32)

			# sample train z
			batch_rgb = rgb_.eval({a:real_as, indexes:indexes_, train:True})
			voxes = []
			for nn in range(12):	# iterate on 12 zs
				v = np.concatenate([batch_rgb[nn],real_as[nn]], -1)
				v = dataset.transformBack(v)
				voxes.append(v)
			saveConcatVoxes2image(voxes, myconfig.vox_prefix+"{0}.train.jpg".format(epoch))

			# sample random z
			for n in range(5):	# iterate on 5 rgbas
				c_image = np.array(real_images[n])
				c_index = np.array(indexes_[n])
				c_a = np.array(real_images[n,:,:,:,3:4])

				# save ground truth
				np.save(myconfig.vox_prefix+"%d-%d.gt.npy"%(epoch, n), dataset.transformBack(c_image))

				# save generated
				rep_a = np.tile(c_a, (batch_size, 1, 1, 1, 1))
				batch_rgb = sampler.eval({a:rep_a, z:random_z, train:False})
				voxes = []
				for nn in range(12):	# iterate on 12 zs
					v = np.concatenate([batch_rgb[nn],c_a], -1)
					v = dataset.transformBack(v)
					voxes.append(v)
				saveConcatVoxes2image(voxes, myconfig.vox_prefix+"{0}-{1}.test.jpg".format(epoch, n))

							


		##################################################
		## save network parameter
		##################################################
		if epoch % save_interval == 0:
			saver.save(sess, myconfig.param_prefix+"{0}.ckpt".format(epoch))
		

		

