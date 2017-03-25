import _init_paths
import tensorflow as tf
import numpy as np
import model
import dataset
import time
import os

import config as myconfig

def prepare_batch_dict(data_dict):
    batch_voxels = data_dict['rgba']        # (n,32,32,32,4)
    batch_z = np.random.uniform(-1, 1, [batch_size, z_size]).astype(np.float32)
    batch_a = batch_voxels[:,:,:,:,3:4]     # (n,32,32,32,1)
    batch_dict = {'rgba':batch_voxels, 'a':batch_a, 'z':batch_z}
    return batch_dict

data = dataset.read()

batch_size = 32
learning_rate = 0.0001
beta1 = 0.5
z_size = 5
save_interval = 50

###  input variables
z = tf.placeholder(tf.float32, [batch_size, z_size])
a = tf.placeholder(tf.float32, [batch_size, 32, 32, 32, 1])
rgba = tf.placeholder(tf.float32, [batch_size, 32, 32, 32, 4])
train = tf.placeholder(tf.bool)

### build models
G = model.Generator(z_size)
D = model.Discriminator()

rgba_ = G(a, z, train)
y_ = D(rgba_, train)
y = D(rgba, train)

label_real = np.zeros([batch_size, 2], dtype=np.float32)
label_fake = np.zeros([batch_size, 2], dtype=np.float32)
label_real[:, 0] = 1
label_fake[:, 1] = 1

loss_G = -tf.reduce_mean(y_)
loss_D = -tf.reduce_mean(y) + tf.reduce_mean(y_)

var_G = [v for v in tf.trainable_variables() if 'g_' in v.name]
var_D = [v for v in tf.trainable_variables() if 'd_' in v.name]

d_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(loss_D, var_list=var_D)
g_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(loss_G, var_list = var_G)

d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in var_D]


print 'var_G'
for v in var_G:
	print v
print 'var_D'
for v in var_D:
	print v

print '\nepoch: ', myconfig.version

print 'loss_csv:', myconfig.loss_csv
print 'vox_prefix:', myconfig.vox_prefix

saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = data.train.num_examples / batch_size
    gen_iterations = 0
    # running
    for epoch in xrange(0, myconfig.ITER_MAX):	
		### train one epoch ### 
        loss_list = {'G':[], 'D':[]}
        for i in xrange(total_batch): 
			# train D
            d_iters = 5
            if gen_iterations % 500 == 0 or gen_iterations < 25:
                 d_iters = 100
            for j in range(0, d_iters):
                batch_dict = prepare_batch_dict(data.train.next_batch(batch_size))
                sess.run(d_clip)
                sess.run(d_rmsprop, feed_dict={a:batch_dict['a'], z:batch_dict['z'], rgba:batch_dict['rgba'], train:True})

			# train G
            batch_dict = prepare_batch_dict(data.train.next_batch(batch_size))
            sess.run(g_rmsprop, feed_dict={a:batch_dict['a'], z:batch_dict['z'], train:True})

			# evaluate
            batch_loss_G = sess.run(loss_G, feed_dict={a:batch_dict['a'], z:batch_dict['z'], train:False})
            batch_loss_D = sess.run(loss_D, feed_dict={a:batch_dict['a'], z:batch_dict['z'], 
                                                        rgba:batch_dict['rgba'], train:False})
            print "{0}, {1}, {2:.8f}, {3:.8f}".format(epoch, i, batch_loss_G, batch_loss_D)
            loss_list['G'].append(batch_loss_G)
            loss_list['D'].append(batch_loss_D)

            gen_iterations +=  1

        ### output losses ###
        loss_G_mean = np.mean(loss_list['G'])
        loss_D_mean = np.mean(loss_list['D'])
        with open(myconfig.loss_csv, 'a') as f:
            msg = "{0}, {1:.8f}, {2:.8f}".format(epoch, loss_G_mean, loss_D_mean)
            print >> f, msg
            print msg

		### output voxels by G every 10 epoches ###
        if epoch%10 != 0:
            continue 
		# 1st ground truth
        np.save(myconfig.vox_prefix+"{0}-sample.npy".format(epoch), 
                    dataset.transformBack(batch_dict['rgba'][0]))

		# generated 
        voxels_ = sess.run(rgba_, feed_dict={a:batch_dict['a'],z:batch_dict['z'],train:False})
        for j, v in enumerate(voxels_[:4]):
            v = v.reshape([32, 32, 32, 4])
            v = dataset.transformBack(v)
            np.save(myconfig.vox_prefix+"{0}-{1}.npy".format(epoch, j), v)

        if epoch % save_interval == 0:
            saver.save(sess, myconfig.param_prefix+"{0}.ckpt".format(epoch))






