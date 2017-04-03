import numpy as np
import collections
import glob
import os
import random
import time

import config

Datasets = collections.namedtuple('Datasets', ['train'])

class Dataset:

	def __init__(self):
		self.name = "real, not jittered"
		print "dataset is ", self.name
		print config.dataset_path

		self.index_in_epoch = 0
		self.examples = np.array(self.read_txt(config.dataset_path))
		self.num_examples = len(self.examples)
		np.random.shuffle(self.examples)

	def read_txt(self, txtFile):
		txtDir = os.path.dirname(txtFile)
		obj_path_list = []
		for obj_path in open(txtFile, 'r'):
			obj_path = obj_path.strip()
			obj_path = os.path.join(txtDir, obj_path)
			obj_path_list.append(obj_path)
		return obj_path_list

	def next_batch(self, batch_size):
		start = self.index_in_epoch
		self.index_in_epoch += batch_size

		if self.index_in_epoch > self.num_examples:
			np.random.shuffle(self.examples)
			start = 0
			self.index_in_epoch = batch_size
			assert batch_size <= self.num_examples

		end = self.index_in_epoch
		return self.read_data(start, end)

	def read_data(self, start, end):
		# 
		batch = {'rgba':[]}
		#s = time.time()
		for fname in self.examples[start:end]:
			if '64.points.npy' in fname:
				points = np.load(fname)
				vox = points2vox(points,64)
			elif '32.points.npy' in fname:
                                points = np.load(fname)
                                vox = points2vox(points,32)
			else:
				vox = np.load(fname)
			#vox = voxJitter(vox)
			data = transformTo(vox)
			batch['rgba'].append(data['rgba'])
		#print 'time ', time.time()-s
		batch['rgba'] = np.array(batch['rgba'])
		return batch


######### dataset <-> network #########

# from dataset to network
# (0,1) -> (-1,1)
def transformTo(vox):
	vox = (vox-0.5)*2

	data = {'rgba':vox}
	return data

# from network to dataset
# rgb: (-1,1) -> (0,1)
# a:   (-1,0) -> 0; (0,1) -> 1
def transformBack(vox):
	vox[:,:,:,3] = (vox[:,:,:,3]>0)
	vox[:,:,:,0:3] = vox[:,:,:,0:3]*0.5+0.5
	return vox


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

######### voxel jitter #########

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


#########  Main  #########

def read():
	train = Dataset()
	return Datasets(train=train)


