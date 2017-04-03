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

def pltPoints(pointname,N):
	start = time.time()
	# load .npy file
	s = time.time()
	points = np.load(pointname) 	
	print 'time ', time.time()-s
	print points.shape
	vox = points2vox(points,N)

	# draw 
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_aspect("equal")
	dim = vox.shape[0]
	ax.set_xlim(0, dim)
	ax.set_ylim(0, dim)
	ax.set_zlim(0, dim)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	xs,ys,zs,rgbs = getPoints(vox)
	ax.scatter(xs,dim-1-ys, dim-1-zs, color=rgbs, s=5)

	print 'time:', time.time()-start
	plt.show()
	

if __name__ == '__main__':
	N = int(sys.argv[1])
	pointname = os.path.abspath(sys.argv[2])
	pltPoints(pointname,N)






