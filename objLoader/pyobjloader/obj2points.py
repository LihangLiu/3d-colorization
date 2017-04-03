# cd path/to/objroot
# usage: python path/to/obj2points.py N file.obj path/to/saved.npy 
#

import sys
import os
import numpy as np
import tinyobjloader as tol

def getPoints(vox):
## xs: (n,1)
## rbgs: (n,1)
        vox_a = vox[:,:,:,3]
        xs,ys,zs = np.nonzero(vox_a)
        rgbs = vox[xs,ys,zs,0:3]
        return xs,ys,zs,rgbs

def obj2vox(filename, N, C):
	grid = tol.LoadObj2Vox(filename,N)
	grid = np.array(grid)
	grid = np.reshape(grid, (N,N,N,C))
	return grid

def obj2points(filename, N, C):
	vox = obj2vox(filename, N, C)
	xs,ys,zs,rgbs = getPoints(vox)
	xs,ys,zs = [np.expand_dims(vs,axis=1) for vs in [xs,ys,zs]]
	points = np.concatenate((xs,ys,zs,rgbs),axis=1)
	return points

if __name__ == '__main__':
	# load obj file and convert to points
	N = int(sys.argv[1])
	filename = os.path.abspath(sys.argv[2])
	C = 4
	points = obj2points(filename,N,C)

	# save to npy file
	npy_path = sys.argv[3]
	np.save(npy_path, points)




