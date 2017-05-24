from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from itertools import product, combinations
import os
import sys
import time
import random
from skimage import io, color

def getPoints(vox):
## xs: (n,1)
## rbgs: (n,1)
	vox_a = vox[:,:,:,3]
	xs,ys,zs = np.nonzero(vox_a)
	rgbs = vox[xs,ys,zs,0:3]
	xs,ys,zs = [np.expand_dims(vs,axis=1) for vs in [xs,ys,zs]]
	points = np.concatenate((xs,ys,zs,rgbs),axis=1)
	return points

def points2vox(points,N):
# points: (n,5)
#	(n,0) -> x
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

def vox2labvox(vox):
	vox[:,:,:,0] = vox[:,:,:,0]*100.0
	vox[:,:,:,1] = vox[:,:,:,1]*115.0
	vox[:,:,:,2] = vox[:,:,:,2]*115.0
	subvox = np.reshape(vox[:,:,:,0:3],[1,-1,3])
	subvox = color.lab2rgb(subvox)
	subvox = np.reshape(subvox,[64,64,64,3])
	vox[:,:,:,0:3] = subvox
	return vox

def getBoundary(vox):
	points = getPoints(vox)
	xs,ys,zs = points[:,0], points[:,1], points[:,2]
	minCoord = np.array([np.min(xs),np.min(ys),np.min(zs)])
	maxCoord = np.array([np.max(xs),np.max(ys),np.max(zs)])
	return minCoord, maxCoord

def cutVox(vox, minCoord, maxCoord):
	min_x, min_y, min_z = minCoord.astype(int)
	max_x, max_y, max_z = maxCoord.astype(int)
	vox = np.array(vox)
	N = vox.shape[0]
	keep_subvox = vox[min_x:max_x,min_y:max_y,min_z:max_z,:]
	points = getPoints(vox)
	points[:,3:6] = 0	# warning
	vox = points2vox(points,N)
	vox[min_x:max_x,min_y:max_y,min_z:max_z,:] = keep_subvox
	return vox

def cutVox(vox):
	vox = np.array(vox)
	N = vox.shape[0]
	points = getPoints(vox)
	xyzs = points[:,0:3]
	sums = np.sum(xyzs, axis=1)
	min_sum, max_sum = np.min(sums), np.max(sums)
	cut_sum = min_sum + (max_sum-min_sum)/2
	sums = (sums>cut_sum)
	points[sums,3:6] = 0	# warning
	vox = points2vox(points,N)
	return vox


def pltPoints(points,N):
	# load .npy file		
	vox = points2vox(points,N)
	labvox = vox2labvox(vox)

	# draw 
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_aspect("equal")
	dim = labvox.shape[0]
	ax.set_xlim(0, dim)
	ax.set_ylim(0, dim)
	ax.set_zlim(0, dim)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	points = getPoints(labvox)
	xs,ys,zs,rgbs = points[:,0], points[:,1], points[:,2], points[:,3:6]
	ax.scatter(xs,dim-1-ys, dim-1-zs, color=rgbs) #, s=5)

	# plt.show()
	

if __name__ == '__main__':
	N = int(sys.argv[1])
	pointname = os.path.abspath(sys.argv[2])
	# load points
	points = np.load(pointname) 
	print points.shape
	# pltPoints(points,N)
	# cut vox
	vox = points2vox(points, N)
	for i in xrange(1):
		new_vox = cutVox(vox)
		points = getPoints(new_vox)
		# pltPoints(points,N)
		np.save(pointname+'.cut-%d.npy'%(i), points)
		print pointname+'.cut-%d.npy'%(i)
	# plt.show()
	






