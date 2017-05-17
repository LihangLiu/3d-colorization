# 
# convert vox to a new obj file with the help of original obj file.
# usage: python vox2obj.py vox.npy oldobj.obj output/path

import sys
from os.path import join
import numpy as np
import random
from skimage import io, color

from objloader import *

def getPoints(vox):
## xs: (n,1)
## rbgs: (n,1)
	vox_a = vox[:,:,:,3]
	xs,ys,zs = np.nonzero(vox_a)
	rgbs = vox[xs,ys,zs,0:3]
	xs,ys,zs = [np.expand_dims(vs,axis=1) for vs in [xs,ys,zs]]
	points = np.concatenate((xs,ys,zs,rgbs),axis=1)
	return points

def getBoundary(vertices, N):
	minCoord = np.array([9999999,9999999,9999999])
	maxCoord = np.array([-9999999,-9999999,-9999999])
	for v in vertices:
		minCoord = np.minimum(minCoord, v)
		maxCoord = np.maximum(maxCoord, v)
	origin = np.array(minCoord)
	scale = np.amax(maxCoord-minCoord)
	scale = (N-1)/scale				# -1: in case of overflow
	return origin, scale

def toGrid(v, origin, scale):
	v = (v-origin)*scale
	return v.astype(np.int32)

def face2grid(v1,v2,v3,origin,scale):
	v1 = np.array(v1)
	v2 = np.array(v2)
	v3 = np.array(v3)
	grid_xyz_list = []
	grid_xyz_list.append(toGrid(v1,origin,scale))
	grid_xyz_list.append(toGrid(v2,origin,scale))
	grid_xyz_list.append(toGrid(v3,origin,scale))
	for i in range(10):
		a = random.uniform(0,1)
		b = random.uniform(0,1-a)
		c_v = v1 + a*(v2-v1) + b*(v3-v1)
		grid_xyz_list.append(toGrid(c_v,origin,scale))

	return grid_xyz_list




if __name__ == '__main__':
	print sys.argv
	if len(sys.argv)!=4:
		print 'usage: python vox2obj.py vox.npy oldobj.obj output/path'
		exit(0)
	# input 
	vox = np.load(sys.argv[1])

	# points to labpoints
	vox[:,:,:,0] = vox[:,:,:,0]*100.0
	vox[:,:,:,1] = vox[:,:,:,1]*115.0
	vox[:,:,:,2] = vox[:,:,:,2]*115.0
	subvox = np.reshape(vox[:,:,:,0:3],[1,-1,3])
	subvox = color.lab2rgb(subvox)
	subvox = np.reshape(subvox,[64,64,64,3])
	vox[:,:,:,0:3] = subvox

	N = vox.shape[0]
	points = getPoints(vox)
	obj = OBJ(sys.argv[2], swapyz=False)
	output_path = sys.argv[3]

	# compose mtl file
	mtl_dict = {}
	with open(join(output_path, 'model_from_vox.mtl'),'w') as f:	
		print >> f, '# obj file converted from vox and the original obj file'
		for point in points:
			x,y,z,r,g,b = point
			mtl_id = 'material_%d_%d_%d'%(x,y,z)
			print >> f, 'newmtl '+mtl_id
			print >> f, 'Kd %f %f %f'%(r,g,b)
			print >> f, 'Ka 0 0 0'
			print >> f, 'Ks 0.4 0.4 0.4'
			print >> f, 'Ke 0 0 0'
			print >> f, 'Ns 10'
			print >> f, 'illum 2'
			print >> f, ''
			mtl_dict[mtl_id] = []

	# get boundary for all vertices
	origin, scale = getBoundary(obj.vertices, N)
	# group face into mtl
	print 'total faces', len(obj.faces)
	count = 0
	for face in obj.faces:
		vertex_ids, normals_ids, tex_ids, mtl_id, group_id = face
		v1 = obj.getVertex(vertex_ids[0])
		v2 = obj.getVertex(vertex_ids[1])
		v3 = obj.getVertex(vertex_ids[2])
		grid_xyz_list = face2grid(v1,v2,v3,origin,scale)
		for x,y,z in grid_xyz_list:
			mtl_id = 'material_%d_%d_%d'%(x,y,z)
			if mtl_id not in mtl_dict:
				# print 'face doesnt map to existing mtl', mtl_id
				count += 1
			else:
				mtl_dict[mtl_id].append((vertex_ids, normals_ids, tex_ids))
				break
	print 'broken faces',count

	# compose obj file
	with open(join(output_path, 'model_from_vox.obj'),'w') as f:
		print >> f, '# obj file converted from vox and the original obj file'
		print >> f, 'mtllib model_from_vox.mtl'
		for v in obj.vertices:
			x,y,z = v
			print >> f, 'v %f %f %f'%(x,y,z)

		for n in obj.normals:
			x,y,z = n
			print >> f, 'vn %f %f %f'%(x,y,z)

		for mtl_id in mtl_dict.keys():
			print >> f, 'g group_'+mtl_id
			print >> f, 'usemtl '+mtl_id
			for vertex_ids, normals_ids, tex_ids in mtl_dict[mtl_id]:
				msg = 'f %d//%d %d//%d %d//%d'%(vertex_ids[0],normals_ids[0],
												vertex_ids[1],normals_ids[1],
												vertex_ids[2],normals_ids[2])
				print >> f, msg
















