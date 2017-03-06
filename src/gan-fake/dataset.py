import numpy as np
import collections
import glob
import os
import random

import config

Datasets = collections.namedtuple('Datasets', ['train'])

class Dataset:

	def __init__(self):
		self.name = "fake, alpha always 1"
		print "dataset is ", self.name

		self.index_in_epoch = 0
		# self.examples = np.array(glob.glob(config.dataset_path))
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
		# create fake dataset:
		# 	the same color for all alpha = 1
		batch = {'rgba':[]}

		for fname in self.examples[start:end]:
			vox = np.load(fname)
			data = transformTo(vox)
			batch['rgba'].append(data['rgba'])

		batch['rgba'] = np.array(batch['rgba'])
		return batch

# from dataset to network
def transformTo(vox):
	vox[:,:,:,3] = 1
	for c in range(3):
		vox[:,:,:,c] = randomColor(vox[:,:,:,c],vox[:,:,:,3])
	vox[:,:,:,3] = (vox[:,:,:,3]-0.5)/0.5

	data = {'rgba':vox}
	return data

def randomColor(vox_c, vox_a):
	c = random.uniform(-1, 1)
	vox_c = vox_c*vox_a
	mask = np.nonzero(vox_c)
	vox_c[mask] = c
	return vox_c

# from network to dataset
def transformBack(vox):
	vox[:,:,:,3] = (vox[:,:,:,3]>0)
	for c in range(3):
		vox[:,:,:,c] = scale(vox[:,:,:,c],vox[:,:,:,3])
	return vox

def scale(vox_c, vox_a):
	vox_c = vox_c*vox_a
	mask = np.nonzero(vox_c)
	vox_c[mask] = vox_c[mask]*0.5+0.5
	return vox_c

def read():
	train = Dataset()
	return Datasets(train=train)


