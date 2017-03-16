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
					ys.append(1-j*delta)
					zs.append(1-k*delta)
					rgbs.append(rgb)
	return xs,ys,zs,rgbs



# load .npy file
vox = np.load(sys.argv[1]) 
print vox.shape
dim = vox.shape[0]
delta = 1.0/dim

# draw cube
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")
r = [0, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
	if np.sum(np.abs(s-e)) == r[1]-r[0]:
		ax.plot3D(*zip(s, e), color="b")

# get data points
xs,ys,zs,rgbs = pointsToDraw(vox)

# draw
start = time.time()
ax.scatter(xs, ys, zs, color=rgbs, s=5)
print 'time:', time.time()-start
plt.savefig('test.png')
plt.show()
exit(0)



