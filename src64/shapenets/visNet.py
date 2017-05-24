import _init_paths
import tensorflow as tf
import numpy as np
import model
import dataset
import time
import os

from append_syn_id import index2name
import config as myconfig

def prepare_batch_dict(data_dict):
    batch_voxels = data_dict['rgba']        # (n,64,64,64,4)
    batch_syn_id = data_dict['syn_id']      # (n,)
    #print batch_syn_id
    batch_label = np.zeros([batch_size, num_syn], dtype=np.float32) # (n,55)
    batch_label[np.array(range(0,batch_size)), batch_syn_id] = 1
    #print batch_label
    batch_dict = {'rgba':batch_voxels, 'label':batch_label, 'syn_id':batch_syn_id}
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

max_y_index = tf.argmax(y,1)        # (n,1)
label_index = tf.argmax(label,1)    # (n,1)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(label,1))   # (n,1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # (1)

print '\nepoch: ', myconfig.version

print 'loss_csv:', myconfig.loss_csv
print 'vox_prefix:', myconfig.vox_prefix

saver = tf.train.Saver(max_to_keep=0)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    if myconfig.preload_model:
        saver.restore(sess, myconfig.preload_model)
        print myconfig.preload_model

    ### test on one epoch ### 
    def print_top_n(syn_accuracy_matrix, n):
        for syn_index in xrange(num_syn):
            c_pred = syn_accuracy_matrix[syn_index,:]
            c_count = np.sum(c_pred)
            c_accuracy = c_pred[syn_index]/c_count
            top_pred_index = c_pred.argsort()[::-1][:n]
            top_pred_value = c_pred[top_pred_index]
            print '%10s ('%(index2name(syn_index)), 
            if c_count == 0:
                print '',
            else:
                for i in xrange(n):
                    if top_pred_value[i] == 0: break
                    print '%10s:%.3f, '%(index2name(top_pred_index[i]),top_pred_value[i]/c_count),
            print ')',
            print '%.3f/%d'%(c_accuracy,c_count)

    syn_accuracy_matrix = np.zeros((num_syn,num_syn))   # 0: label, 1: y
    accuracy_list = []
    for i in xrange(test_data.num_examples/batch_size):
        batch_dict = prepare_batch_dict(test_data.next_batch(batch_size))
        feed_dict = prepare_feed_dict(batch_dict, rgba, label, train,False)
        # accuracy of every syn
        test_accuracy = sess.run(accuracy,feed_dict=feed_dict)
        test_y_index, test_label_index = sess.run([max_y_index,label_index],feed_dict=feed_dict)
        for j in xrange(batch_size):
		syn_accuracy_matrix[test_label_index[j],test_y_index[j]] += 1
	# syn_accuracy_matrix[test_label_index,test_y_index] += 1

        accuracy_list.append(test_accuracy)
        print i, test_accuracy
        print_top_n(syn_accuracy_matrix, 3)

    # tbd

    accuracy_mean = np.mean(accuracy_list)
    print accuracy_mean



