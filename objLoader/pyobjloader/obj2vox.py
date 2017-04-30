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

def mapTexture(texture_img, uv):
  assert len(texture_img.shape)==3, texture_img.shape
  w,h,d = texture_img.shape
  u,v = uv
  u = int(u%1*w)%w
  v = int(v%1*h)%h
  rgb = texture_img[u,v,:]/255.0
  # print u,v,rgb
  # a = input('press')
  # print texture_img
  return rgb

def getBoundary(vertices, N):
  minCoord = np.array([9999999,9999999,9999999])
  maxCoord = np.array([-9999999,-9999999,-9999999])
  for v in vertices:
    minCoord = np.minimum(minCoord, v)
    maxCoord = np.maximum(maxCoord, v)
  origin = np.array(minCoord)
  scale = np.amax(maxCoord-minCoord)
  scale = (N-1)/scale       # -1: in case of overflow
  return origin, scale

def areaOfTriangle(v1, v2, v3):
  AB = v2-v1;
  AC = v3-v1;
  res = np.power(AB[1]*AC[2] - AB[2]*AC[1], 2);
  res += np.power(AB[0]*AC[2] - AB[2]*AC[0], 2);
  res += np.power(AB[0]*AC[2] - AB[1]*AC[0], 2);
  res = np.sqrt(res)/2;
  return res;

def toGrid(v, origin, scale):
  v = (v-origin)*scale
  return v.astype(np.int32)

def face2grid(v1,v2,v3,origin,scale):
  v1 = np.array(v1)
  v2 = np.array(v2)
  v3 = np.array(v3)
  grid_xyz_list = []
  sampleRate = 1+10*int(areaOfTriangle(v1*scale,v2*scale,v3*scale));
  for i in range(sampleRate):
    a = random.uniform(0,1)
    b = random.uniform(0,1-a)
    c_v = v1 + a*(v2-v1) + b*(v3-v1)
    grid_xyz_list.append(toGrid(c_v,origin,scale))

  return grid_xyz_list

def obj2vox(obj, N):
  # get boundary for all vertices
  origin, scale = getBoundary(obj.vertices, N)

  # map faces to grid, sum up rgb and a first and then average
  grid = np.zeros((N,N,N,4))
  for face in obj.faces:
    vertex_ids, normals_ids, tex_ids, mtl_id, group_id = face
    # Kd value
    rgb = obj.mtl[mtl_id]['Kd']
    # check uv 
    if tex_ids[0] != 0:
      uv1 = obj.getTexcoords(tex_ids[0])
      uv2 = obj.getTexcoords(tex_ids[1])
      uv3 = obj.getTexcoords(tex_ids[2])
      uv = np.mean([uv1,uv2,uv3], axis=0)
      if 'image' in obj.mtl[mtl_id]:
        image = obj.mtl[mtl_id]['image']
        rgb = mapTexture(image,uv)
    # sample face 
    v1 = obj.getVertex(vertex_ids[0])
    v2 = obj.getVertex(vertex_ids[1])
    v3 = obj.getVertex(vertex_ids[2])
    grid_xyz_list = face2grid(v1,v2,v3,origin,scale)
    for x,y,z in grid_xyz_list:
      grid[x,y,z,0:3] += rgb[0:3]     # tbd, ValueError: operands could not be broadcast together with shapes (3,) (4,) (3,)
      grid[x,y,z,3] += 1

  # average all rgb and a 
  for x in range(N):
    for y in range(N):
      for z in range(N):
        if grid[x,y,z,3] != 0:
          grid[x,y,z,0:3] /= grid[x,y,z,3]
          # print grid[x,y,z,0:3]
          grid[x,y,z,3] = 1

  return grid


if __name__ == '__main__':
  print sys.argv
  if len(sys.argv)!=4:
    print 'usage: python obj2vox.py N obj.obj vox.npy'
    exit(0)
  # input 
  start = time.time()
  N = int(sys.argv[1])
  obj = OBJ(sys.argv[2], swapyz=False)
  vox_path = sys.argv[3]
  # print 'total faces', len(obj.faces)
  # print 'textures'
  # for mtl_id in obj.mtl:
  #   if 'image' in obj.mtl[mtl_id]:
  #     print obj.mtl[mtl_id]['map_Kd']

  # convert obj to vox
  vox = obj2vox(obj,N)  

  # save to npy file
  np.save(vox_path, grid)
  print 'generated', vox_path
  print 'time used', time.time()-start







  







