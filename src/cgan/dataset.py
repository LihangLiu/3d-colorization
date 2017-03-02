import numpy as np
import collections
import glob
import os

import config

Datasets = collections.namedtuple('Datasets', ['train'])

class Dataset:

	def __init__(self):
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
		# rgb: [-1,1]
		# a: {0,1}
		data = []

		for fname in self.examples[start:end]:
		   # data.append(util.read_binvox(fname))
			vox = np.load(fname)
			vox[:,:,:,:3] = (vox[:,:,:,:3]-0.5)/0.5
			data.append(vox)

		return np.array(data)

def read():
	train = Dataset()
	return Datasets(train=train)


