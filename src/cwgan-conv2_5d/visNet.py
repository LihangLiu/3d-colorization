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
lr = 5e-5
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

d_rmsprop = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss_D, var_list=var_D)
g_rmsprop = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss_G, var_list = var_G)

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
print 'param_prefix', myconfig.param_prefix

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# restore saved model
saver = tf.train.Saver()
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
epoch = 2000         # specify the wanted model
model_path = myconfig.param_prefix+"{0}.ckpt".format(epoch)
saver.restore(sess, model_path)

# fetch variables
batch_dict = prepare_batch_dict(data.train.next_batch(batch_size))
fetches_G = sess.run(loss_G, feed_dict={a:batch_dict['a'], z:batch_dict['z'], train:False})
fetches_D = sess.run(loss_D, feed_dict={a:batch_dict['a'], z:batch_dict['z'], 
                                                        rgba:batch_dict['rgba'], train:False})

# test
batch_dict = prepare_batch_dict(data.train.next_batch(batch_size))
tmp_a = batch_dict['a']
tmp_rgba = batch_dict['rgba']
np.save("testgt.npy", dataset.transformBack(tmp_rgba[0]))
fetches_G = sess.run(rgba_, feed_dict={a:tmp_a, z:batch_dict['z'], train:False})
print 'z:', batch_dict['z'][0,:]
print fetches_G[0,:,:,0,0]
np.save("test1.npy", dataset.transformBack(fetches_G[0]))

batch_dict = prepare_batch_dict(data.train.next_batch(batch_size))
fetches_G = sess.run(rgba_, feed_dict={a:tmp_a, z:batch_dict['z'], train:False})
print 'z:', batch_dict['z'][0,:]
print fetches_G[0,:,:,0,0]
np.save("test2.npy",dataset.transformBack(fetches_G[0]))
