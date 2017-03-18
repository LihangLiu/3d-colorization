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

def pointsToDraw(vox):
	dim = vox.shape[0]
	vox_r = vox[:,:,:,0]
	vox_g = vox[:,:,:,1]
	vox_b = vox[:,:,:,2]
	vox_a = vox[:,:,:,3]

	mask = np.nonzero(vox_a)
	xs = mask[0].tolist()
	ys = (dim-mask[1]).tolist()
	zs = (dim-mask[2]).tolist()

	rgbs = np.zeros((len(xs),3))
	rgbs[:,0] = vox_r[mask]
	rgbs[:,1] = vox_g[mask]
	rgbs[:,2] = vox_b[mask]
	rgbs = np.clip(rgbs,0,1)
	rgbs = rgbs.tolist()

	return xs,ys,zs,rgbs

def vox2image(voxname,imname):
	start = time.time()
	# load .npy file
	vox = np.load(voxname) 	

	# draw 
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_aspect("equal")
	xs,ys,zs,rgbs = pointsToDraw(vox)
	ax.scatter(xs, ys, zs, color=rgbs, s=5)
	plt.savefig(imname)

	print 'time:', time.time()-start
	

if __name__ == '__main__':
	voxname = os.path.abspath(sys.argv[1])
	imname = os.path.abspath(sys.argv[2])

	# load obj file and convert to vox
	vox2image(voxname,imname)








