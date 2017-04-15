import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np
from scipy.misc import imread
from os.path import dirname
import os
import sys
import time

import random


class Dataset:

	def __init__(self, dataset_path):
		self.name = "real,"
		print "dataset is ", self.name
		print dataset_path

		self.index_in_epoch = 0
		self.examples = np.array(self.read_txt(dataset_path))
		self.num_examples = len(self.examples)
		print 'total size: ', self.num_examples
		self.shuffle_index = range(0, len(self.examples))
		c = list(zip(self.examples, self.shuffle_index))
		np.random.shuffle(c)
		self.examples, self.shuffle_index = zip(*c)
		

	def read_txt(self, txtFile):
		txtDir = os.path.dirname(txtFile)
		obj_path_list = []
		for obj_path in open(txtFile, 'r'):
			obj_path = obj_path.strip().split()[0]
			obj_path = os.path.join(txtDir, obj_path)
			obj_path_list.append(obj_path)
		return obj_path_list

	def next_batch(self, batch_size):
		start = self.index_in_epoch
		self.index_in_epoch += batch_size

		if self.index_in_epoch > self.num_examples:
			c = list(zip(self.examples, self.shuffle_index))
			np.random.shuffle(c)
			self.examples, self.shuffle_index = zip(*c)
			#np.random.shuffle(self.examples)
			start = 0
			self.index_in_epoch = batch_size
			assert batch_size <= self.num_examples

		end = self.index_in_epoch
		return self.read_data(start, end)

	def read_data(self, start, end):
		batch = {'rgba':[]}
		for fname in self.examples[start:end]:
			if '64.points.npy' in fname:
				points = np.load(fname)
				vox = points2vox(points,64)
			elif '32.points.npy' in fname:
				points = np.load(fname)
				vox = points2vox(points,32)
			else:
				vox = np.load(fname)
			vox = transformTo(vox)
			batch['rgba'].append(vox)

		batch['rgba'] = np.array(batch['rgba'])
		batch['index'] = np.array(self.shuffle_index[start:end])
		return batch

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
# points: (n,5)
#       (n,0) -> x
#       (n,1) -> y
#       (n,2) -> z
#       (n,3:6) -> rgb
	xs = points[:,0].astype(int)
	ys = points[:,1].astype(int)
	zs = points[:,2].astype(int)
	rgb = points[:,3:6]
	vox = np.zeros((N,N,N,4))
	vox[xs,ys,zs,0:3] = rgb
	vox[xs,ys,zs,3] = 1
	return vox

# alpha channel: non-0 as 1
# rgb channels: assume to between (0,1)
#		

def getPoints(vox):
## xs: (n,1)
## rbgs: (n,1)
	vox_a = vox[:,:,:,3]
	xs,ys,zs = np.nonzero(vox_a)
	rgbs = vox[xs,ys,zs,0:3]
	return xs,ys,zs,rgbs

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




