import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np
from scipy.misc import imread
from os.path import dirname, basename
import os
import sys
import time
import random
from skimage import io, color


class Dataset:

	def __init__(self, dataset_path, max_num=None, if_shuffle=True):
		self.name = "real, not jittered"
		self.dataset_path = dataset_path
		self.if_shuffle = if_shuffle
		print "dataset is ", self.name
		print dataset_path

		self.index_in_epoch = 0
		if max_num is None:
			self.examples = np.array(self.read_txt(dataset_path))
		else:
			self.examples = np.array(self.read_txt(dataset_path))[:max_num]
		self.num_examples = len(self.examples)
		print 'dataset size: ', self.num_examples
		if self.if_shuffle: np.random.shuffle(self.examples)

	def fetch_all(self):
		points_list = []
		self.path_list = []
		self.examples = np.array(self.read_txt(self.dataset_path))
		for fname, syn_id in self.examples:
			self.path_list.append(fname)
			if 'points' in fname: 	# if points
				points = np.load(fname)
			else:		
				print "voxels are large, fetch_all not suggested", fname
				exit(0)
			points_list.append(points)

		# print all_points
		return points_list

	def read_txt(self, txtFile):
		txtDir = os.path.dirname(txtFile)
		obj_path_list = []
		for line in open(txtFile, 'r'):		# obj_path [syn_id] -> (obj_path,syn_id=-1)
			line = line.strip().split()
			syn_id = '-1'
			if len(line) == 2:
				syn_id = line[1]
			obj_path = line[0]
			obj_path = os.path.join(txtDir, obj_path)
			obj_path_list.append((obj_path, syn_id))
		return obj_path_list

	def next_batch(self, batch_size):
		start = self.index_in_epoch
		self.index_in_epoch += batch_size

		if self.index_in_epoch > self.num_examples:
			if self.if_shuffle: np.random.shuffle(self.examples)
			start = 0
			self.index_in_epoch = batch_size
			assert batch_size <= self.num_examples

		end = self.index_in_epoch
		return self.read_data(start, end)

	SUFFIX_LABPOINTS = '64.labpoints.npy'
	SUFFIX_POINTS = '64.points.npy'

	def read_data(self, start, end):
		batch = {'rgba':[], 'syn_id':[]}
		for fname, syn_id in self.examples[start:end]:
			if self.SUFFIX_POINTS in fname or self.SUFFIX_LABPOINTS in fname: 	# if points
				points = np.load(fname)
				vox = points2vox(points,64)
			elif '32.points.npy' in fname:					
				points = np.load(fname)
				vox = points2vox(points,32)
			else:														# if vox
				vox = np.load(fname)
			#vox = voxJitter(vox)
			if not self.SUFFIX_LABPOINTS in fname:	# norm if not lab
				vox = transformTo(vox)
			batch['rgba'].append(vox)
			batch['syn_id'].append(int(syn_id))

		batch['rgba'] = np.array(batch['rgba'])
		batch['syn_id'] = np.array(batch['syn_id'])
		return batch


#######################
## dataset <-> network
#######################

# from dataset to network
# (0,1) -> (-1,1)
# input: voxel or batch_voxel
def transformTo(vox):
	new_vox = np.array(vox)
	shape = new_vox.shape
	assert len(shape)==4 or len(shape)==5
	assert shape[-1] == 4
	new_vox = (new_vox-0.5)*2
	return new_vox

# from network to dataset
# rgb: (-1,1) -> (0,1)
# a:   (-1,0) -> 0; (0,1) -> 1
# input: voxel or batch_voxel
def transformBack(vox):
	new_vox = np.array(vox)
	shape = new_vox.shape
	assert len(shape)==4 or len(shape)==5
	assert shape[-1] == 4
	if len(shape)==4:
		new_vox[:,:,:,3] = (new_vox[:,:,:,3]>0)
		new_vox[:,:,:,0:3] = new_vox[:,:,:,0:3]*0.5+0.5
	else:
		new_vox[:,:,:,:,3] = (new_vox[:,:,:,:,3]>0)
		new_vox[:,:,:,:,0:3] = new_vox[:,:,:,:,0:3]*0.5+0.5
	return new_vox


def points2vox(points,N):
# points: (n,6)
#       (n,0) -> x
#       (n,1) -> y
#       (n,2) -> z
#       (n,3:6) -> rgb or lab
	xs = points[:,0].astype(int)
	ys = points[:,1].astype(int)
	zs = points[:,2].astype(int)
	rgb = points[:,3:6]
	vox = np.zeros((N,N,N,4))
	vox[xs,ys,zs,0:3] = rgb
	vox[xs,ys,zs,3] = 1
	return vox


#######################
## rgbvox to image
#######################

def concatenateImages(imname_list,out_imname):
	N = len(imname_list)
	W = 4
	H = int((N-1)/W)+1
	for i,imname in enumerate(imname_list):
		plt.subplot(H, W, i+1)
		img = imread(imname)
		plt.imshow(img)
	plt.savefig(out_imname,dpi=1000)
	plt.close()
	

def vox2image(vox,imname):
	# draw 
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_aspect("equal")
	dim = vox.shape[0]
	ax.set_xlim(0, dim)
	ax.set_ylim(0, dim)
	ax.set_zlim(0, dim)
	xs,ys,zs,rgbs = getPoints(vox)
	ax.scatter(xs,dim-1-ys, dim-1-zs, color=rgbs) #, s=5)

	plt.savefig(imname)
	plt.close(fig)

def saveConcatVoxes2image(voxes, imname):
	sub_names = []
	basename = os.path.basename(imname)
	for i,vox in enumerate(voxes): 
		# print i,vox
		sub_name = "tmp/%s-tmp-%d.jpg"%(basename, i)
		vox2image(vox, sub_name)
		sub_names.append(sub_name)
	concatenateImages(sub_names, imname)
	print imname
	for name in sub_names:
		os.remove(name)

def labvox2vox(vox):
	vox[:,:,:,0] = vox[:,:,:,0]*100.0
	vox[:,:,:,1] = vox[:,:,:,1]*115.0
	vox[:,:,:,2] = vox[:,:,:,2]*115.0
	subvox = np.reshape(vox[:,:,:,0:3],[1,-1,3])
	subvox = color.lab2rgb(np.array(subvox,np.float64))	# float32->float64, bug otherwise
	subvox = np.reshape(subvox,[64,64,64,3])
	vox[:,:,:,0:3] = subvox
	return vox	

def vox2labvox(vox):
	grid = np.array(vox)
	labs = color.rgb2lab(grid[:,:,:,0:3])
	labs[:,:,:,0] = labs[:,:,:,0]/100.0
	labs[:,:,:,1] = labs[:,:,:,1]/115.0
	labs[:,:,:,2] = labs[:,:,:,2]/115.0
	grid[:,:,:,0:3] = labs
	return grid

#######################
## voxel jitter 
#######################

def getBoundary(vox):
	xs,ys,zs,_ = getPoints(vox)
	minCoord = np.array([np.min(xs),np.min(ys),np.min(zs)])
	maxCoord = np.array([np.max(xs),np.max(ys),np.max(zs)])
	return minCoord, maxCoord

def getPoints(vox):
## xs: (n,1)
## rbgs: (n,1)
	vox_a = vox[:,:,:,3]
	xs,ys,zs = np.nonzero(vox_a)
	rgbs = vox[xs,ys,zs,0:3]
	return xs,ys,zs,rgbs

def shiftVox(vox, del_xyz=[0,0,0]):
	xs,ys,zs,rgbs = getPoints(vox)

	xs += del_xyz[0]
	ys += del_xyz[1]
	zs += del_xyz[2]
	
	new_vox = np.zeros(vox.shape)
	new_vox[xs,ys,zs,0:3] = rgbs
	new_vox[xs,ys,zs,3] = 1
	return new_vox

def flipVox(vox, flip_xyz=[False,False,False]):
	dim = vox.shape[0]
	xs,ys,zs,rgbs = getPoints(vox)

	if flip_xyz[0]:
		xs = dim-1-xs
	if flip_xyz[1]:
		ys = dim-1-ys
	if flip_xyz[2]:
		zs = dim-1-zs
	
	new_vox = np.zeros(vox.shape)
	new_vox[xs,ys,zs,0:3] = rgbs
	new_vox[xs,ys,zs,3] = 1
	return new_vox

def rotateVox(vox, rotate_xyz=[0,1,2]):
	dim = vox.shape[0]
	xs,ys,zs,rgbs = getPoints(vox)

	xyzs = [xs,ys,zs]
	xs = xyzs[rotate_xyz[0]]
	ys = xyzs[rotate_xyz[1]]
	zs = xyzs[rotate_xyz[2]]
	
	new_vox = np.zeros(vox.shape)
	new_vox[xs,ys,zs,0:3] = rgbs
	new_vox[xs,ys,zs,3] = 1
	return new_vox

def sampleFlip():
	return [random.randint(0,1) for i in range(3)]

def sampleShift(vox):
	minCoord, maxCoord = getBoundary(vox)
	minDel = -minCoord
	maxDel = np.array(vox.shape[0:3])-1-maxCoord
	return [random.randint(minDel[i],maxDel[i]) for i in range(3)]

def sampleRotate():
	res = [0,1,2]
	random.shuffle(res)
	return res

def voxJitter(vox):
	vox = shiftVox(vox, sampleShift(vox))
	#vox = flipVox(vox, sampleFlip())
	#vox = rotateVox(vox, sampleRotate())
	return vox



