import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np
from os.path import dirname
import os
import sys
import time


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
	vox = np.load(voxname) 	

	# draw 
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_aspect("equal")
	dim = vox.shape[0]
	ax.set_xlim(0, dim)
	ax.set_ylim(0, dim)
	ax.set_zlim(0, dim)
	xs,ys,zs,rgbs = getPoints(vox)
	ax.scatter(xs, ys, zs, color=rgbs, s=5)

	print 'time:', time.time()-start
	plt.savefig(imname)
	

if __name__ == '__main__':
	voxname = os.path.abspath(sys.argv[1])
	imname = os.path.abspath(sys.argv[2])

	# load obj file and convert to vox
	vox2image(voxname,imname)























