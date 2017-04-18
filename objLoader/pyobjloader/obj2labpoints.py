# cd path/to/objroot
# usage: python path/to/obj2labpoints.py N file.obj path/to/saved.npy 
#

import sys
import os
import numpy as np
import tinyobjloader as tol
import time
from skimage import io, color

def getPoints(vox):
## xs: (n,1)
## rbgs: (n,1)
        vox_a = vox[:,:,:,3]
        xs,ys,zs = np.nonzero(vox_a)
        labs = vox[xs,ys,zs,0:3]
        return xs,ys,zs,labs

def obj2labvox(filename, N, C):
	grid = tol.LoadObj2Vox(filename,N)
	grid = np.array(grid)
	grid = np.reshape(grid, (N,N,N,C))
	labs = color.rgb2lab(grid[:,:,:,0:3])
	labs[:,:,:,0] = labs[:,:,:,0]/100.0
	labs[:,:,:,1] = labs[:,:,:,1]/115.0
	labs[:,:,:,2] = labs[:,:,:,2]/115.0
	grid[:,:,:,0:3] = labs
	return grid

def obj2labpoints(filename, N, C):
	vox = obj2labvox(filename, N, C)
	xs,ys,zs,labs = getPoints(vox)
	xs,ys,zs = [np.expand_dims(vs,axis=1) for vs in [xs,ys,zs]]
	points = np.concatenate((xs,ys,zs,labs),axis=1)
	return points

if __name__ == '__main__':
	# load obj file and convert to points
	s = time.time()
	N = int(sys.argv[1])
	filename = os.path.abspath(sys.argv[2])
	C = 4
	points = obj2labpoints(filename,N,C)

	# save to npy file
	npy_path = sys.argv[3]
	np.save(npy_path, points)

	print "time: ", time.time()-s




