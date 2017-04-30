# 
# convert vox to a new obj file with the help of original obj file.
# usage: python vox2obj.py vox.npy oldobj.obj output/path

import sys
from os.path import join
import numpy as np
import random
import time
from scipy.misc import imread
from skimage import io, color

from objloader import *
from obj2vox import obj2vox
from obj2points import vox2points

def vox2labvox(vox):
  labs = color.rgb2lab(vox[:,:,:,0:3])
  labs[:,:,:,0] = labs[:,:,:,0]/100.0
  labs[:,:,:,1] = labs[:,:,:,1]/115.0
  labs[:,:,:,2] = labs[:,:,:,2]/115.0
  vox[:,:,:,0:3] = labs
  return vox

def obj2labpoints(obj,N):
  vox = obj2vox(obj, N)
  labvox = vox2labvox(vox)
  labpoints = vox2points(labvox)
  return labpoints

if __name__ == '__main__':
  # print sys.argv
  if len(sys.argv)!=4:
    print 'usage: python obj2vox.py N obj.obj labpoints.npy'
    exit(0)
  # input 
  start = time.time()
  N = int(sys.argv[1])
  obj = OBJ(sys.argv[2], swapyz=False)
  labpoints_path = sys.argv[3]
  # print 'total faces', len(obj.faces)
  # print 'textures'
  # for mtl_id in obj.mtl:
  #   if 'image' in obj.mtl[mtl_id]:
  #     print obj.mtl[mtl_id]['map_Kd']

  # convert obj to vox
  labpoints = obj2labpoints(obj,N)

  # save to npy file
  np.save(labpoints_path, labpoints)
  # print 'generated', labpoints_path
  print 'time used', time.time()-start







  







