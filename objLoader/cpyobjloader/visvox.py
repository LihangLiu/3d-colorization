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


def pltVox(voxname):
	start = time.time()
	# load .npy file
	s = time.time()
	vox = np.load(voxname)
	print 'load time ', time.time()-s 	
	print vox.shape

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
	ax.scatter(xs,dim-1-ys, dim-1-zs, color=rgbs) #, s=5)

	print 'time:', time.time()-start
	plt.show()
	

if __name__ == '__main__':
	voxname = os.path.abspath(sys.argv[1])
	pltVox(voxname)






