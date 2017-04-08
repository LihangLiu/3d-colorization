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

print '\nepoch: ', myconfig.version

print 'loss_csv:', myconfig.loss_csv
print 'vox_prefix:', myconfig.vox_prefix

saver = tf.train.Saver(max_to_keep=0)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    if myconfig.preload_mode:
        saver.restore(sess, myconfig.preload_mode)
    total_batch = train_data.num_examples / batch_size
    # running
    for epoch in xrange(myconfig.ITER_MIN, myconfig.ITER_MAX):  
        loss_list = {'loss_D':[], 'train_accuracy':[], 'test_accuracy':[]}
        ### train on one epoch ### 
        for i in xrange(total_batch): 
            batch_dict = prepare_batch_dict(train_data.next_batch(batch_size))
            feed_dict = prepare_feed_dict(batch_dict, rgba, label, train,True)

            # evaluate
            train_loss_D,train_accuracy = sess.run([loss_D,accuracy], feed_dict=feed_dict)
            loss_list['loss_D'].append(train_loss_D)
            loss_list['train_accuracy'].append(train_accuracy)
            if i%10 == 0:
                with open(myconfig.log_txt, 'a') as f:
                    msg = "%d %d %.6f %.6f\n"% (epoch, i, train_loss_D, train_accuracy)
                    print msg
                    f.write(msg)

            # train
            sess.run(opt_D, feed_dict=feed_dict)

        ### test on one epoch ### 
        for i in xrange(test_data.num_examples/batch_size):
            batch_dict = prepare_batch_dict(test_data.next_batch(batch_size))
            feed_dict = prepare_feed_dict(batch_dict, rgba, label, train,False)
            test_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            loss_list['test_accuracy'].append(test_accuracy)

        ### output losses ###
        loss_D_mean = np.mean(loss_list['loss_D'])
        train_accuracy_mean = np.mean(loss_list['train_accuracy'])
        test_accuracy_mean = np.mean(loss_list['test_accuracy'])
        with open(myconfig.loss_csv, 'a') as f:
            msg = "%d %.6f %.6f %.6f"%(epoch,loss_D_mean,train_accuracy_mean,test_accuracy_mean)
            print >> f, msg
            print msg

        if epoch % save_interval == 0:
            saver.save(sess, myconfig.param_prefix+"{0}.ckpt".format(epoch))







