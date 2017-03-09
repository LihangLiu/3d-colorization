from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from itertools import product, combinations
from os.path import dirname
import sys
import time

def pointsToDraw(vox):
	xs = []
	ys = []
	zs = []
	rgbs = []
	dim = vox.shape[0]
	for i in range(dim):
        	for j in range(dim):
                	for k in range(dim):
                        	if(vox[i,j,k,3]!=0):
                                	rgb = [vox[i,j,k,0], vox[i,j,k,1], vox[i,j,k,2]]
					xs.append(i*delta)
					ys.append(j*delta)
					zs.append(k*delta)
					rgbs.append(rgb)
	return xs,ys,zs,rgbs
#                               print rgb
                           #    if rgb[0]<0:
                            #            continue
                             #   ax.scatter([i*delta], [1-j*delta], [1-k*delta], color=rgb, s=5)


#
vox = np.load(sys.argv[1]) 

# downsample
dim = vox.shape[0]
#R = [0,1]
#vox_new = np.zeros((dim/2,dim/2,dim/2,4))
#vox_count = np.zeros((dim/2,dim/2,dim/2))
#for rx in R:
#	for ry in R:
#		for rz in R:
#			vox_a = vox[rx::2,ry::2,rz::2,3]
#			# vox_a = np.expand_dims(vox_a, -1)
#			for c in range(3):
#				vox_new[:,:,:,c] += vox_a*vox[rx::2,ry::2,rz::2,c]
#			vox_new[:,:,:,3] = np.maximum(vox_new[:,:,:,3], vox_a)
#			vox_count += vox_a
#for c in range(3):
#	vox_new[:,:,:,c] = vox_new[:,:,:,c]/(vox_count+0.00001)
#vox = vox_new
print vox.shape

dim = vox.shape[0]
delta = 1.0/dim

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

# draw cube
r = [0, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
	if np.sum(np.abs(s-e)) == r[1]-r[0]:
		ax.plot3D(*zip(s, e), color="b")


xs,ys,zs,rgbs = pointsToDraw(vox)

start = time.time()
ax.scatter(xs, ys, zs, color=rgbs, s=5)
print 'time:', time.time()-start
plt.show()
exit(0)

# draw a point
for i in range(dim):
	for j in range(dim):
		for k in range(dim):
			if(vox[i,j,k,3]!=0 and i<3):
				rgb = [vox[i,j,k,0], vox[i,j,k,1], vox[i,j,k,2]]
#				print rgb
				if rgb[0]<0:
					continue
				ax.scatter([i*delta], [1-j*delta], [1-k*delta], color=rgb, s=5)
	
plt.show()
exit(0)



