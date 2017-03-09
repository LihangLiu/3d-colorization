import sys
import os
import numpy as np
import tinyobjloader as tol


def obj2vox(filename, N, C):
	grid = tol.LoadObj2Vox(filename,N)
	grid = np.array(grid)
	grid = np.reshape(grid, (N,N,N,C))
	return grid

if __name__ == '__main__':
	filename = os.path.abspath(sys.argv[1])

	# load obj file and convert to vox
	N = 32
	C = 4
	grid = obj2vox(filename,N,C)

	# save to npy file
	if len(sys.argv) > 1:
		npy_path = sys.argv[2]
		np.save(npy_path, grid)




