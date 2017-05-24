#import config
import numpy as np
import tensorflow as tf
import os
import dataset
#config.dataset_path = '../data/npy_list.02958343.32.voxels.txt.train' 
#categories = ['02958343','04379243', '02691156', '03001627', '04256520', '04090263','03636649','04530566','02828884','03691459','02933112','03211117','04401088','02924116','02808440','03467517','03325088', '03046257','03991062']
#categories = ['02958343']
#categories = ['02691156', '03001627']
#categories = ['02876657', '03790512', '03948459']
#categories = ['02954340']
#categories = ['all7']
categories = ['02958343', '02691156', '03001627', '02876657', '03790512', '03948459', '02954340', 'all7']
def read_txt(txtFile):
    txtDir = os.path.dirname(txtFile)
    obj_path_list = []
    for obj_path in open(txtFile, 'r'):
        obj_path = obj_path.strip().split()[0]
            #obj_path = os.path.join(txtDir, obj_pat
        #obj_path = '/scratch/cluster/yzp12/dataset/'+obj_path
        obj_path = obj_path[:-16]+'npy'
        obj_path_list.append(obj_path)
    return obj_path_list

N = len(categories)
config = tf.ConfigProto()
for i in range(N):

    tf.reset_default_graph()
    g = tf.Graph()
    cat = categories[i]
    dataset_path = '../data/lists/lab_npy_list.{0}.64.points.txt.train'.format(cat)
    examples = np.array(read_txt(dataset_path))
    batch = []
    #batch_size = min(len(examples), 1000)
    if cat == '02958343':
        examples = examples[:1000]
    elif cat != 'all7':
        examples = examples[:300]

    #batch_size = min(len(examples), 300)
    batch_size = len(examples)
    for i in range(batch_size):
        line = examples[i]
        fname = line.strip()[:-15] + '32.points.npy'
        points = np.load(fname)
        vox = dataset.points2vox(points,32)
        batch.append(vox[:,:,:,3:4])
    real_images = np.array(batch).astype('float32')
    z_size = 20
    with g.as_default():
        flattened_images = tf.get_variable('pca', [batch_size, 32, 32, 32, 1], tf.float32, tf.random_normal_initializer(stddev = 0.02))
        assign_place = flattened_images.assign(real_images)
        flattened_images = tf.reshape(flattened_images, [batch_size, -1])
        flattened_images = flattened_images - tf.reduce_mean(flattened_images, axis=0)
        cov_matrix = tf.matmul(flattened_images, tf.transpose(flattened_images))
##get the eigenvalues of covariance matrix
        [e,v] = tf.self_adjoint_eig(cov_matrix)
    ##get the top k eigenvectors
        [topkval, topk_idx] = tf.nn.top_k(e, 1)


    sess = tf.Session(graph=g)
    sess.run(assign_place)
    top_var = sess.run(topkval)
   
    #np.save('../data/dist_matrix_l2_{0}.npy'.format(cat), sess.run(dist_matrix))
    
    print 'finish cat: {0}, top 1 eigen value: {1}'.format(cat, top_var)
print "success"
