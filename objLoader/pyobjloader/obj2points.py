# 
# convert vox to a new obj file with the help of original obj file.
# usage: python vox2obj.py vox.npy oldobj.obj output/path

import sys
from os.path import join
import numpy as np
import random
import time
from scipy.misc import imread

from objloader import *
from obj2vox import obj2vox

def vox2points(vox):
## xs: (n,1)
## rbgs: (n,1)
  vox_a = vox[:,:,:,3]
  xs,ys,zs = np.nonzero(vox_a)
  rgbs = vox[xs,ys,zs,0:3]
  xs,ys,zs = [np.expand_dims(vs,axis=1) for vs in [xs,ys,zs]]
  points = np.concatenate((xs,ys,zs,rgbs),axis=1)
  return points

def obj2points(obj,N):
  vox = obj2vox(obj,N)
  points = vox2points(vox)
  return points

if __name__ == '__main__':
  # print sys.argv
  if len(sys.argv)!=4:
    print 'usage: python obj2vox.py N obj.obj points.npy'
    exit(0)
  # input 
  start = time.time()
  N = int(sys.argv[1])
  obj = OBJ(sys.argv[2], swapyz=False)
  points_path = sys.argv[3]
  # print 'total faces', len(obj.faces)
  # print 'textures'
  # for mtl_id in obj.mtl:
  #   if 'image' in obj.mtl[mtl_id]:
  #     print obj.mtl[mtl_id]['map_Kd']

  # convert obj to vox
  points = obj2points(obj, N)

  # save to npy file
  np.save(points_path, points)
  # print 'generated', points_path
  print 'time used', time.time()-start







  







