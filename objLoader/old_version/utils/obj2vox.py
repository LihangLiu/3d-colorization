import sys
from os.path import dirname
import numpy as np
import random

from objloader import *

# the vertex coordinates are assumed to be between [-0.5, 0.5]

def inGridXYZ(v1):
  v1 = (v1+0.5)/delta
  return v1.astype(int)

def getInsectGridXYZs(v1, v2, v3):
  v1 = np.array(v1)
  v2 = np.array(v2)
  v3 = np.array(v3)

  grid_xyz_list = []
  grid_xyz_list.append(inGridXYZ(v1))
  grid_xyz_list.append(inGridXYZ(v2))
  grid_xyz_list.append(inGridXYZ(v3))
  
  for i in range(10):
    a = random.uniform(0,1)
    b = random.uniform(0,1-a)
    c_v = v1 + a*(v2-v1) + b*(v3-v1)
    grid_xyz_list.append(inGridXYZ(c_v))

  return grid_xyz_list



obj = OBJ(sys.argv[1], swapyz=True)

# discretize the grid as n*n*n
n = 64
delta = 1.0/n
#grid = np.zeros((n,n,n), dtype = np.int)
grid = np.zeros((n,n,n,4), dtype = np.int)


for face in obj.faces:
    vertex_ids, normals_ids, tex_ids, mtl_id = face
    v1 = obj.getVertex(vertex_ids[0])
    v2 = obj.getVertex(vertex_ids[1])
    v3 = obj.getVertex(vertex_ids[2])
    for x,y,z in getInsectGridXYZs(v1, v2, v3):
      grid[x,y,z] += 1


np.save('voxel.npy', grid)





