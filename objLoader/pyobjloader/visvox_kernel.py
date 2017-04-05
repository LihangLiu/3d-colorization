import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from itertools import product, combinations
import os
import sys
import time
import random

# alpha channel: non-0 as 1
# rgb channels: assume to between (0,1)
#		otherwise clipped to be (0,1)


def getPoints(vox):
## xs: (n,1)
## rbgs: (n,1)
	vox_a = vox[:,:,:,3]
	xs,ys,zs = np.nonzero(vox_a)
	rgbs = vox[xs,ys,zs,0:3]
	return xs,ys,zs,rgbs

def downsample(vox):
	return vox[::2,::2,::2,:]

def maxpooling(vox):
	new_vox = vox[::2,::2,::2,:]
	delta = [0,1]
	for x in delta:
		for y in delta:
			for z in delta:
				new_vox = np.maximum(new_vox,vox[x::2,y::2,z::2,:])
	return new_vox

def patch(vox,x,y,z,m):
	return vox[x:x+m,y:y+m,z:z+m,:]

def savePatches(vox,imname):
	m = 4
	for x in range(0,vox.shape[0],m):
		for y in range(0,vox.shape[1],m):
			for z in range(0,vox.shape[2],m):
				vox_patch = patch(vox,x,y,z,m)
				saveVox(vox_patch,imname+'.%d_%d_%d.jpg'%(x,y,z))

def saveVox(vox, imname):
	start = time.time()
	#print vox.shape
	# draw 
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_aspect("equal")
	dim = vox.shape[0]
	ax.set_xlim(0, dim)
	ax.set_ylim(0, dim)
	ax.set_zlim(0, dim)
	xs,ys,zs,rgbs = getPoints(vox)
	ax.scatter(xs,ys, zs, color=rgbs) #, s=5)
	print "total points:",xs.shape[0]

	#print 'time:', time.time()-start
	if xs.shape[0] != 0:
		count_list.append(xs.shape[0])
		plt.savefig(imname)
	plt.close(fig)
	

if __name__ == '__main__':
	voxname = os.path.abspath(sys.argv[1])
	# load .npy file
	vox = np.load(voxname)


	# downsample
	#vox = downsample(vox)
	#vox = maxpooling(vox)
	# vox = patch(vox)

	#saveVox(vox,voxname+'.jpg')
	count_list = []
	savePatches(vox,voxname)
	print "average points ", np.sum(count_list)/len(count_list)






