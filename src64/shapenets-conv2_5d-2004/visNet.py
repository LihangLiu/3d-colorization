import _init_paths
import tensorflow as tf
import numpy as np
import model
import dataset
import time
import os

import config as myconfig

def prepare_batch_dict(data_dict):
    batch_voxels = data_dict['rgba']        # (n,64,64,64,4)
    batch_syn_id = data_dict['syn_id']      # (n,)
    #print batch_syn_id
    batch_label = np.zeros([batch_size, num_syn], dtype=np.float32) # (n,55)
    batch_label[np.array(range(0,batch_size)), batch_syn_id] = 1
    #print batch_label
    batch_dict = {'rgba':batch_voxels, 'label': batch_label}
    return batch_dict

def prepare_feed_dict(batch_dict, rgba, label, train, flag):
    fetch_dict = {rgba:batch_dict['rgba'], label:batch_dict['label'], train:flag}
    return fetch_dict

train_data = dataset.Dataset(myconfig.train_dataset_path)
test_data = dataset.Dataset(myconfig.test_dataset_path)

batch_size = 32
learning_rate = 0.0001
beta1 = 0.5
save_interval = myconfig.save_interval
num_syn = 55

###  input variables
rgba = tf.placeholder(tf.float32, [batch_size, 64, 64, 64, 4])
label = tf.placeholder(tf.float32, shape=[batch_size, 55])  #each row is a one-hot 

train = tf.placeholder(tf.bool)

### build models
D = model.Discriminator()
y = D(rgba, train)
loss_D = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y))
opt_D = tf.train.AdamOptimizer(learning_rate, beta1).minimize(loss_D)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(label,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

var_D = [v for v in tf.trainable_variables() if 'd_' in v.name]

print '\nepoch: ', myconfig.version

print 'loss_csv:', myconfig.loss_csv
print 'vox_prefix:', myconfig.vox_prefix

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# restore saved model
saver = tf.train.Saver()
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
model_path = "../../outputs/params/params2004_50.ckpt"
saver.restore(sess, model_path)

# fetch variables
batch_dict = prepare_batch_dict(test_data.next_batch(batch_size))
feed_dict = prepare_feed_dict(batch_dict, rgba, label, train,False)
var_D = sess.run(var_D, feed_dict=feed_dict)
for var in var_D:
	if var.shape == ():
		print var
#print var_D

sess.close()


