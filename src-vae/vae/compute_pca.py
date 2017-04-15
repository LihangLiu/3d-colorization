import numpy as np
import tensorflow as tf
import os
import sys
import dataset
def read_txt(txtFile):
    txtDir = os.path.dirname(txtFile)
    obj_path_list = []
    for obj_path in open(txtFile, 'r'):
        obj_path = obj_path.strip().split()[0]
        obj_path = os.path.join(txtDir, obj_path)
        obj_path_list.append(obj_path)
    return obj_path_list

txt_path = sys.argv[1]
examples = np.array(read_txt(txt_path))
#examples = examples[1:100]
batch_size = len(examples)
z_size = 20
batch = {'rgba':[]}

for fname in examples:
    vox = np.load(fname)
    data = dataset.transformTo(vox)
    batch['rgba'].append(data['rgba'])

batch['rgba'] = np.array(batch['rgba'])


real_images = batch['rgba']

#flattened_images = tf.Variable([0.0])
place = tf.placeholder(tf.float32, shape=(batch_size, 32, 32, 32, 4))
#set_x = flattened_images.assign(place)
flattened_images = tf.Variable(place)
sess = tf.Session()
sess.run(tf.initialize_all_variables(), feed_dict={place:real_images})

flattened_images = tf.reshape(flattened_images, [batch_size, -1])
flattened_images = flattened_images - tf.reduce_mean(flattened_images, axis=0)
cov_matrix = tf.matmul(flattened_images, tf.transpose(flattened_images))
##get the eigenvalues of covariance matrix
[e,v] = tf.self_adjoint_eig(cov_matrix)
    ##get the top k eigenvectors
[topkval, topk_idx] = tf.nn.top_k(e, z_size)
 
topkv = tf.gather(tf.transpose(v), topk_idx)
topkv_ori = tf.matmul(topkv, flattened_images)
proj_images = tf.matmul(flattened_images, tf.transpose(topkv_ori))
root_n = tf.constant(np.sqrt(batch_size),'float32')
proj_images = tf.scalar_mul(root_n,tf.nn.l2_normalize(proj_images, 0, epsilon=1e-12, name=None))
np.save('pca_vectors.npy', sess.run(proj_images))
print "success"

