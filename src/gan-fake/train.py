import tensorflow as tf
import numpy as np
import model
import dataset
import time


data = dataset.read()

batch_size = 32
learning_rate = 0.0001
beta1 = 0.5
z_size = 50
save_interval = 1

x = tf.placeholder(tf.float32, [batch_size, 32, 32, 32, 4])
z = tf.placeholder(tf.float32, [batch_size, z_size])
train = tf.placeholder(tf.bool)

G = model.Generator(z_size)
D = model.Discriminator()

x_ = G(z)
y_ = D(x_, train)

y = D(x, train)

label_real = np.zeros([batch_size, 2], dtype=np.float32)
label_fake = np.zeros([batch_size, 2], dtype=np.float32)
label_real[:, 0] = 1
label_fake[:, 1] = 1

loss_G = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=tf.constant(label_real)))
loss_D = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=tf.constant(label_fake)))
loss_D += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=tf.constant(label_real)))

var_G = [v for v in tf.trainable_variables() if 'g_' in v.name]
var_D = [v for v in tf.trainable_variables() if 'd_' in v.name]

opt_G = tf.train.AdamOptimizer(learning_rate, beta1).minimize(loss_G, var_list=var_G)
opt_D = tf.train.AdamOptimizer(learning_rate, beta1).minimize(loss_D, var_list=var_D)

saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())
	total_batch = data.train.num_examples / batch_size

	# running for 500 epoches
	for epoch in xrange(1, 500):	
		# train one epoch
		loss_list = {'G':[], 'D':[]}
		for i in xrange(total_batch): 
			# input
			voxels = data.train.next_batch(batch_size)
			batch_z = np.random.uniform(-1, 1, [batch_size, z_size]).astype(np.float32)

			# forward-backward
			sess.run(opt_G, feed_dict={z:batch_z, train:True})
			if i%5 == 0:
				sess.run(opt_D, feed_dict={x:voxels, z:batch_z, train:True})

			# evaluate
			batch_loss_G = sess.run(loss_G, feed_dict={z:batch_z, train:False})
			batch_loss_D = sess.run(loss_D, feed_dict={x:voxels, z:batch_z, train:False})
			loss_list['G'].append(batch_loss_G)
			loss_list['D'].append(batch_loss_D)

		# print loss
		loss_G_mean = np.mean(loss_list['G'])
		loss_D_mean = np.mean(loss_list['D'])
		with open("outputs/voxels/loss_7.csv", 'a') as f:
			msg = "{0}, {1:.8f}, {2:.8f}".format(epoch, loss_G_mean, loss_D_mean)
			print >> f, msg
			print msg


		# sample outputs
		batch_z = np.random.uniform(-1, 1, [batch_size, z_size]).astype(np.float32)
		voxels = sess.run(x_, feed_dict={z:batch_z})

		for j, v in enumerate(voxels[:5]):
			v = v.reshape([32, 32, 32, 4])
			np.save("outputs/voxels/epoch7_{0}-{1}.npy".format(epoch, j), v)

#		if epoch % save_interval == 0:
#			saver.save(sess, "outputs/params/epoch6_{0}.ckpt".format(epoch))

