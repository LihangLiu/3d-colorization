from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from itertools import product, combinations
from os.path import dirname
import sys


def normlize(vox_c, vox_a):
	vox = vox_c*vox_a
	mask = np.nonzero(vox)
	comp = vox[mask]
	print np.mean(comp), np.std(comp)
	norm = (comp-np.mean(comp))/np.std(comp)
	vox[mask] = norm
	return vox	

def getNonzero(vox):
	return vox[np.nonzero(vox)]

def scale(vox):
	min_ = np.amin(vox)-0.2
	max_ = np.amax(vox)+0.1
	vox = (vox-min_)/(max_-min_)
	return vox

#
vox = np.load(sys.argv[1]) 
# normlize
#for c in range(3):
#	vox[:,:,:,c] = normlize(vox[:,:,:,c],vox[:,:,:,3])

# histogram
#plt.hist(getNonzero(vox[:,:,:,0]), bins=np.arange(0,1,0.1))
#plt.show()
#exit(0)

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


# convert vox
#vox[:,:,:,0:3] = (vox[:,:,:,0:3] + 1)/2
for c in range(3):
       vox[:,:,:,c] = scale(vox[:,:,:,c])
vox[:,:,:,3] = (vox[:,:,:,3]>0.5)

# draw a point
for i in range(dim):
    for j in range(dim):
        for k in range(dim):
            if(vox[i,j,k,3]!=0):
            	rgb = [vox[i,j,k,0], vox[i,j,k,1], vox[i,j,k,2]]
		#print rgb
                ax.scatter([1-i*delta], [1-j*delta], [1-k*delta], color=rgb, s=5)
		ax.scatter([i*delta], [j*delta], [k*delta], color=rgb, s=5)

plt.show()
exit(0)



