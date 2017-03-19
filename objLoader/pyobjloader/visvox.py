from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from itertools import product, combinations
from os.path import dirname
import sys
import time

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
	rgbs = rgbs.tolist()

	# xs = []
	# ys = []
	# zs = []
	# rgbs = []
	# dim = vox.shape[0]
	# for i in range(dim):
	# 	for j in range(dim):
	# 		for k in range(dim):
	# 			if(vox[i,j,k,3]!=0):
	# 				rgb = [vox[i,j,k,0], vox[i,j,k,1], vox[i,j,k,2]]
	# 				xs.append(i*delta)
	# 				ys.append(1-j*delta)
	# 				zs.append(1-k*delta)
	# 				rgbs.append(rgb)
	return xs,ys,zs,rgbs


start = time.time()
# load .npy file
vox = np.load(sys.argv[1]) 
print vox.shape
dim = vox.shape[0]

# draw cube
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")
ax.set_xlim(0, dim)
ax.set_ylim(0, dim)
ax.set_zlim(0, dim)
#r = [0, 1]
#for s, e in combinations(np.array(list(product(r, r, r))), 2):
#	if np.sum(np.abs(s-e)) == r[1]-r[0]:
#		ax.plot3D(*zip(s, e), color="b")

# get data points
xs,ys,zs,rgbs = pointsToDraw(vox)

# draw
ax.scatter(xs, ys, zs, color=rgbs, s=5)
print 'time:', time.time()-start

plt.show()




