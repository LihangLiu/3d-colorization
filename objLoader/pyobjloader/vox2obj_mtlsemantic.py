# 
# convert vox to a new obj file with the help of original obj file.
# usage: python vox2obj_mtlsemantic.py vox.npy oldobj.obj output/path
# vox.npy: vox or points

import sys
from os.path import join
import numpy as np
import random
from skimage import io, color

from objloader import *


def points2vox(points,N):
# points: (n,5)
#	(n,0) -> x
#       (n,1) -> y
#       (n,2) -> z
#       (n,3:6) -> rgb
	xs = points[:,0].astype(int)
	ys = points[:,1].astype(int)
	zs = points[:,2].astype(int)
	rgb = points[:,3:6]
	vox = np.zeros((N,N,N,4))
	vox[xs,ys,zs,0:3] = rgb
	vox[xs,ys,zs,3] = 1
	return vox

# mtlkd_dict: {mtl_id, Kd_values}
# Kd_values: (float,float,float)
# only change Kd value
def write2mtl(old_mtl_name, new_mtl_name, mtlkd_dict):
	mtl_id = None
	with open(new_mtl_name,'w') as new_f:
		for line in open(old_mtl_name, "r"):
			if line.startswith('#'): 
				new_f.write(line)
				continue
			values = line.split()
			if not values: 
				new_f.write(line)
				continue

			if values[0] == 'newmtl':
				mtl_id = values[1]
			elif mtl_id is None:
				raise ValueError, "mtl file doesn't start with newmtl stmt"
			elif values[0] == 'Kd':
				if mtl_id in mtlkd_dict:
					new_kd = mtlkd_dict[mtl_id]
					line = 'Kd %f %f %f\n'%(new_kd[0],new_kd[1],new_kd[2])
				else:
					print 'mtl id not found in obj', mtl_id
			new_f.write(line)
	print new_mtl_name, 'created'

# only change the mtl file path
def write2obj(old_obj_name, new_obj_name, relative_new_mtl_name):
	with open(new_obj_name,'w') as new_f:
		for line in open(old_obj_name, "r"):
			if line.startswith('mtllib'): 
				line = 'mtllib '+new_mtl_name
			new_f.write(line)
	print new_obj_name, 'created'

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




if __name__ == '__main__':
	print sys.argv
	if len(sys.argv)!=4:
		print 'usage: python vox2obj.py vox.npy oldobj.obj output/path'
		exit(0)
	# input 
	voxpath = sys.argv[1]
	if '.64.points.npy' in voxpath:
		points = np.load(voxpath)
		vox = points2vox(points, 64)
	else:
		vox = np.load(voxpath)

	vox[:,:,:,0] = vox[:,:,:,0]*100.0
	vox[:,:,:,1] = vox[:,:,:,1]*115.0
	vox[:,:,:,2] = vox[:,:,:,2]*115.0
	subvox = np.reshape(vox[:,:,:,0:3],[1,-1,3])
	subvox = color.lab2rgb(subvox)
	subvox = np.reshape(subvox,[64,64,64,3])
	vox[:,:,:,0:3] = subvox
	
	N = vox.shape[0]
	old_obj_name = sys.argv[2]
	obj = OBJ(sys.argv[2], swapyz=False)
	output_path = sys.argv[3]

	# get boundary for all vertices
	origin, scale = getBoundary(obj.vertices, N)

	# group faces by mtl_id and xyz in grid world, {mtl_id:{(x,y,z):rgb}}
	mtlkd_dict = {}
	for face in obj.faces:
		vertex_ids, normals_ids, tex_ids, mtl_id, group_id = face
		v1 = obj.getVertex(vertex_ids[0])
		v2 = obj.getVertex(vertex_ids[1])
		v3 = obj.getVertex(vertex_ids[2])
		grid_xyz_list = face2grid(v1,v2,v3,origin,scale)
		for x,y,z in grid_xyz_list:
			if vox[x,y,z,3] == 0:
				continue
			rgb = vox[x,y,z,0:3]
			if mtl_id in mtlkd_dict:
				if (x,y,z) not in mtlkd_dict[mtl_id]:
					mtlkd_dict[mtl_id][(x,y,z)] = rgb
			else:
				mtlkd_dict[mtl_id] = {}
				mtlkd_dict[mtl_id][(x,y,z)] = rgb

	# calculate average rgb for each mtl_id
	# with open('log.txt','w') as f:
	# 	print >> f, mtlkd_dict
	for mtl_id in mtlkd_dict:
		rgbs = np.array(mtlkd_dict[mtl_id].values())
		print mtl_id, rgbs.shape
		mtlkd_dict[mtl_id] = np.mean(rgbs, axis=0)
		# mtlkd_dict[mtl_id] = np.median(rgbs, axis=0)


	# compose new mtl file
	new_mtl_name = join(output_path,'model_from_vox.mtl')
	write2mtl(obj.mtl_path, new_mtl_name,mtlkd_dict)

	# compose new obj file
	new_obj_name = join(output_path,'model_from_vox.obj')
	write2obj(old_obj_name, new_obj_name, 'model_from_vox.mtl')
	















