import matplotlib 
matplotlib.use('Agg')

import _init_paths
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import multiprocessing
import time

from dataset import Dataset, points2vox
from vislabpoints import pltPoints

# points: (M,6)
# 6-th: alpha channel, 0 or 1
def f_similarity_all(all_points):
	N = len(all_points)
	vox_mat = np.zeros((N,64*64*64))
	for i,points in enumerate(all_points):
		print i
		vox = points2vox(points,64)
		vox_mat[i,:] = np.reshape(vox[:,:,:,3],[64*64*64])
	sim_mat = np.matmul(vox_mat,np.transpose(vox_mat))
	return sim_mat

# each point (x,y,z,r,g,b) in point1 find it's nearest neighbor in point0
# 
def color_mapping(points_src, points_dst):
	vox_src = points2vox(points_src,64)
	vox_dst = points2vox(points_dst,64)
	new_points = np.array(points_dst)
	for i,(x,y,z,r,g,b) in enumerate(new_points):
		x,y,z = int(x),int(y),int(z)
		if vox_src[x,y,z,3] == 1:
			new_points[i,3:6] = vox_src[x,y,z,0:3]
		else:
			xyzs_src = np.array(points_src[:,0:3])
			xyz_dst = np.array([x,y,z])
			dists = np.sum(np.square(xyzs_src-xyz_dst),axis=1)
			minj = np.argmin(dists)
			new_points[i,3:6] = points_src[minj,3:6]
	return new_points

def thread_func(sim_mat,all_points,all_paths,mini,maxi):
	for i in xrange(mini,maxi):
		print i
		# make direcotory in point i
		c_path_i = os.path.dirname(str(all_paths[i]))
		c_subpath_i = os.path.join(c_path_i,'mapped_labpoints')
		if not os.path.exists(c_subpath_i):
			os.makedirs(c_subpath_i)
		else:
			if len(os.listdir(c_subpath_i)) > 10:		# warning
				print 'exists'
				continue
			# print 'exists'
			# continue
		print 'path i', c_subpath_i

		# retrived the most similar shapes
		c_sim = sim_mat[i,:]
		top_sim_index = c_sim.argsort()[::-1][:20]
		start = time.time()
		for j in top_sim_index:
			# print "similarity:", c_sim[j]
			new_points = color_mapping(all_points[i],all_points[j])

			# save to point i
			new_path = os.path.join(c_subpath_i, 'mapped_points_%d.npy'%(j))
			np.save(new_path, new_points)

			# save to point j
			c_path_j = os.path.dirname(str(all_paths[j]))
			c_subpath_j = os.path.join(c_path_j,'bemapped_labpoints')
			if not os.path.exists(c_subpath_j):
				os.makedirs(c_subpath_j)
			# print 'path j', c_subpath_j
			new_path_j = os.path.join(c_subpath_j, 'bemapped_points_%d.npy'%(i))
			np.save(new_path_j, new_points)

		print 'time used:', time.time()-start


def main():
	if len(sys.argv) != 3:
		print 'usage: python color_propagation.py dataset_path_txt similarity_matrix_cache.npy'
		exit(0)

	#################
	# load similarity cache
	#################
	sim_mat = None
	cache_path = sys.argv[2]
	if os.path.exists(cache_path):
		sim_mat = np.load(cache_path)
		print 'load', cache_path

	#################
	# load all points
	#################
	dataset_path = sys.argv[1]
	print 'dataset path', dataset_path
	dataset = Dataset(dataset_path)
	all_points = dataset.fetch_all()
	all_paths = dataset.path_list
	N = len(all_points)
	print 'num of points', N

	#######################
	# cal similarity matrix
	#######################
	if sim_mat is None:
		sim_mat = f_similarity_all(all_points)
		np.save(cache_path, sim_mat)
		print 'save', cache_path

	###################################
	# propagate color to similar shapes
	###################################
	N = min(2000,N) 	# warning, first 2000
	THREAD_NUM = 9
	try:
		batch_size = N/THREAD_NUM
		for i in xrange(THREAD_NUM+1):
			mini,maxi = i*batch_size,(i+1)*batch_size
			maxi = min(maxi,N)
			print 'min-max %d %d'%(mini,maxi)
			p = multiprocessing.Process(target=thread_func, args=(sim_mat,all_points,all_paths,mini,maxi))			
			p.start()
	except:
		print "Error: unable to start thread"

	# while True:
	# 	pass


			

if __name__ == '__main__':
	main()









