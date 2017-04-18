import _init_paths
import tensorflow as tf
import numpy as np
import model
import dataset
import time
import os

import config as myconfig

def prepare_batch_dict(data_dict):
	batch_voxels = data_dict['rgba']		# (n,64,64,64,4)
	batch_z = np.random.uniform(-1, 1, [batch_size, z_size]).astype(np.float32)
	batch_a = batch_voxels[:,:,:,:,3:4]	 # (n,64,64,64,1)
	batch_dict = {'rgba':batch_voxels, 'a':batch_a, 'z':batch_z}
	return batch_dict

def prepare_feed_dict(batch_dict, rgba, a, z, k_t, k_t_, train, flag):
	fetch_dict = {a:batch_dict['a'], z:batch_dict['z'],rgba:batch_dict['rgba'], 
					k_t:min(max(k_t_, 0), 1), train:flag}
	return fetch_dict

train_data = dataset.Dataset(myconfig.train_dataset_path)

##################################################
## set parameter
##################################################
batch_size = 32
learning_rate = 5e-5
beta1 = 0.5
z_size = 5
save_interval = myconfig.save_interval
sample_interval = 10


##################################################.
## build graph
##################################################
z = tf.placeholder(tf.float32, [batch_size, z_size])
a = tf.placeholder(tf.float32, [batch_size, 64, 64, 64, 1])
rgba = tf.placeholder(tf.float32, [batch_size, 64, 64, 64, 4])
train = tf.placeholder(tf.bool)
k_t = tf.placeholder(tf.float32, shape=[])

### build models
G = model.Generator(z_size)
D = model.Discriminator(z_size)

rgba_ = G(a, z, train)
rgba_out_ = D(rgba_, train)
rgba_out = D(rgba, train)

loss_D,loss_G,k_tp,convergence_measure = model.loss(rgba, rgba_out, rgba_, rgba_out_, k_t=k_t)

var_G = [v for v in tf.trainable_variables() if 'g_' in v.name]
var_D = [v for v in tf.trainable_variables() if 'd_' in v.name]

opt_g = tf.train.AdamOptimizer(learning_rate,beta1).minimize(loss_G, var_list = var_G)
opt_d = tf.train.AdamOptimizer(learning_rate,beta1).minimize(loss_D, var_list=var_D)

print 'var_G'
for v in var_G:
	print v
print 'var_D'
for v in var_D:
	print v

print '\nepoch: ', myconfig.version

print 'loss_csv:', myconfig.loss_csv
print 'vox_prefix:', myconfig.vox_prefix


##################################################.
## start session
##################################################
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())

	##################################################
	## check preload model
	##################################################
	saver = tf.train.Saver()
	if myconfig.preload_model:
		print 'load model: ', myconfig.preload_model
		saver.restore(sess, myconfig.preload_model)

	# ##################################################
	# ## train
	# ##################################################	  
	# total_batch = train_data.num_examples / batch_size
	# k_t_ = 0  # We initialise with k_t = 0 as in the paper.
	# for epoch in xrange(myconfig.ITER_MIN, myconfig.ITER_MAX):  
	# 	### train one epoch ### 
	# 	loss_list = {'G':[], 'D':[], 'convergence_measure':[]}
	# 	for i in xrange(total_batch): 
	# 		# train G and D
	# 		batch_dict = prepare_batch_dict(train_data.next_batch(batch_size))
	# 		feed_dict = prepare_feed_dict(batch_dict, rgba, a, z, k_t,k_t_, train,True)
	# 		_,_,b_loss_G,b_loss_D, k_t_,c_m = sess.run([opt_g,opt_d,loss_G,loss_D,k_tp,convergence_measure], feed_dict=feed_dict)

	# 		# evaluate
	# 		loss_list['G'].append(b_loss_G)
	# 		loss_list['D'].append(b_loss_D)
	# 		loss_list['convergence_measure'].append(c_m)
	# 		with open(myconfig.log_txt, 'a') as f:
	# 			msg = "%d %d %.6f %.6f %.6f %.6f"%(epoch, i, b_loss_G, b_loss_D, c_m, k_t_)
	# 			print >> f, msg
 #                print msg


	# 	### output losses ###
	# 	loss_G_mean = np.mean(loss_list['G'])
	# 	loss_D_mean = np.mean(loss_list['D'])
	# 	conv_mea_mean = np.mean(loss_list['convergence_measure'])
	# 	with open(myconfig.loss_csv, 'a') as f:
	# 		msg = "%d %.6f %.6f %.6f"%(epoch,loss_G_mean,loss_D_mean,conv_mea_mean)
	# 		print >> f, msg
	# 		print msg 

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

	### train data
	batch_dict = prepare_batch_dict(train_data.next_batch(batch_size))

	# save ground-truth
	saveConcatVoxes2image(dataset.transformBack(np.array(batch_dict['rgba'][0:12])), 
							"test.rgba.jpg")

	# save generated
	k_t_ = 0
	feed_dict = prepare_feed_dict(batch_dict, rgba, a, z, k_t,k_t_, train,False)
	b_rgba_out, b_rgba_, b_rgba_out_ = sess.run([rgba_out, rgba_, rgba_out_], feed_dict=feed_dict)
	saveConcatVoxes2image(dataset.transformBack(np.array(b_rgba_out[0:12])), 
							"test.rgba_out.jpg")
	saveConcatVoxes2image(dataset.transformBack(np.array(b_rgba_[0:12])), 
							"test.rgba_.jpg")
	saveConcatVoxes2image(dataset.transformBack(np.array(b_rgba_out_[0:12])), 
							"test.rgba_out_.jpg")

	
