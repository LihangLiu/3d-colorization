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

def prepare_feed_dict(batch_dict, rgba, rgb, a, indexes, train, flag):
	fetch_dict = {rgba:batch_dict['rgba'], rgb:batch_dict['rgb'], a:batch_dict['a'],indexes:batch_dict['index'], train:flag}
	return fetch_dict

def prepare_fix_shape_feed_dict(batch_dict, rgb, a, indexes, train, flag):
	batch_rgb = np.array(batch_dict['rgb'])
	batch_index = np.array(batch_dict['index'])
	# shuffle index and rgb
	c = list(zip(batch_rgb, batch_index))
	np.random.shuffle(c)
	batch_rgb, batch_index = zip(*c)
	fetch_dict = {rgb:np.array(batch_rgb), a:batch_dict['a'],indexes:np.array(batch_index), train:flag}
	return fetch_dict

train_data = dataset.Dataset(myconfig.train_dataset_path)
test_data = dataset.Dataset(myconfig.test_dataset_path)

if __name__ == '__main__':

	##################################################
	## set parameter
	##################################################
	batch_size = 32 # for 
	num_train = train_data.num_examples
	learning_rate = 5e-5
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
	rgba = tf.placeholder(tf.float32, [batch_size, 64, 64, 64, 4])
	train = tf.placeholder(tf.bool)
	indexes = tf.placeholder(tf.int32, [batch_size,])
	with tf.variable_scope('G_z'):
		all_z = tf.get_variable("prior_z", [num_train, z_size], tf.float32, tf.random_normal_initializer(stddev=1))

	z = tf.placeholder(tf.float32, [batch_size, z_size])

	### vae ###
	# train graph
	G = model.Generator(z_size)
	rgb_, rgba_ = G.train(a, indexes, all_z, train)
	loss_vae = tf.reduce_mean(tf.abs(rgb-rgb_))
	# sample graph
	sample_rgb_, sample_rgba_ = G.sample(a, z, train)

	### gan ###
	D = model.Discriminator()
	y_ = D(rgba_, train)
	y = D(rgba, train)
	label_real = np.zeros([batch_size, 2], dtype=np.float32)
	label_fake = np.zeros([batch_size, 2], dtype=np.float32)
	label_real[:, 0] = 1
	label_fake[:, 1] = 1

	loss_G = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=tf.constant(label_real)))
	loss_D = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=tf.constant(label_fake)))
	loss_D += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=tf.constant(label_real)))

	var_G = [v for v in tf.trainable_variables() if 'g_' in v.name]
	var_D = [v for v in tf.trainable_variables() if 'd_' in v.name]

	opt_vae = tf.train.AdamOptimizer(learning_rate, beta1).minimize(loss_vae, var_list=var_G)
	opt_D = tf.train.AdamOptimizer(learning_rate, beta1).minimize(loss_D, var_list=var_D)
	opt_G = tf.train.AdamOptimizer(learning_rate, beta1).minimize(loss_G, var_list = var_G)

	print 'var_G'
	for v in var_G:
		print v
	print 'var_D'
	for v in var_D:
		print v
	print '\nepoch: ', myconfig.version
	print 'loss_csv:', myconfig.loss_csv
	print 'vox_prefix:', myconfig.vox_prefix

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
		gen_iterations = 0
		for epoch in xrange(myconfig.ITER_MIN, myconfig.ITER_MAX):
			loss_list = {'vae':[], 'G':[], 'D':[]}
			for i in xrange(total_batch):				
				# ###############
				# ## train vae
				# ###############
				# # update Generator		
				# batch_dict = prepare_batch_dict(train_data.next_batch(batch_size))				
				# feed_dict = prepare_feed_dict(batch_dict, rgba, rgb, a, indexes, train,True)				
				# sess.run(opt_vae, feed_dict=feed_dict)		

				# update fix shape regularization
				# feed_dict = prepare_fix_shape_feed_dict(batch_dict, rgb, a, indexes, train,True)
				# sess.run(opt_vae, feed_dict=feed_dict)

				#####################
				## train wgan and vae
				#####################			
				batch_dict = prepare_batch_dict(train_data.next_batch(batch_size))
				feed_dict = prepare_feed_dict(batch_dict, rgba, rgb, a, indexes, train,True)
				sess.run([opt_G,opt_vae,opt_D], feed_dict=feed_dict)

				###############
				## evaluate
				###############				
				batch_dict = prepare_batch_dict(train_data.next_batch(batch_size))
				feed_dict = prepare_feed_dict(batch_dict, rgba, rgb, a, indexes, train,False)
				batch_loss_list = sess.run([loss_vae, loss_G, loss_D], feed_dict=feed_dict)
				loss_list['vae'].append(batch_loss_list[0])
				loss_list['G'].append(batch_loss_list[1])
				loss_list['D'].append(batch_loss_list[2])
				with open(myconfig.log_txt, 'a') as f:
					msg = "%d %d "%(epoch,i) + str(batch_loss_list)
					print >> f, msg
					print msg								
				# output gradients
				if i%50 == 0:
					grad = tf.gradients(loss_vae, G.W['dh5'])[0]
					print 'grad vae: \n', grad.eval(feed_dict)[0][0][0]
					grad = tf.gradients(loss_G, G.W['dh5'])[0]
					print 'grad G: \n', grad.eval(feed_dict)[0][0][0]
					grad = tf.gradients(loss_D, G.W['dh5'])[0]
					print 'grad D: \n', grad.eval(feed_dict)[0][0][0]				

				# used for wgan
				gen_iterations +=  1

			with open(myconfig.loss_csv, 'a') as f:
				loss_vae_mean = np.mean(loss_list['vae'])
				loss_G_mean = np.mean(loss_list['G'])
				loss_D_mean = np.mean(loss_list['D'])
				f.write("{0}, {1:.8f}, {2:.8f}, {3:.8f}\n".format(epoch, loss_vae_mean, loss_G_mean, loss_D_mean))


			##################################################
			## draw samples on train and test dataset
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
				feed_dict = prepare_feed_dict(batch_dict, rgba, rgb, a, indexes, train,False)
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
			


