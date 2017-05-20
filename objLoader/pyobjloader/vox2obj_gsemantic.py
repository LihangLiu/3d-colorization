# 
# convert vox to a new obj file with the help of original obj file.
# usage: python vox2obj_gsemantic.py vox.npy oldobj.obj output/path
# vox.npy: vox or points

import sys
from os.path import join, split
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

# mtlkd_dict: {(mtl_id,group_id): Kd_values}
# Kd_values: (float,float,float)
# change Kd value and new mtl according to (mtl_id,group_id)
# mtl: obj.mtl
def write2mtl(new_mtl_name, mtl, mtlkd_dict):
	with open(new_mtl_name,'w') as f:	
		print >> f, '# obj file converted from vox and the original obj file\n'
		for key in mtlkd_dict:
			mtl_id,group_id = key
			new_kd = mtlkd_dict[key]
			# write newmtl
			print >> f, 'newmtl %s-%s'%(mtl_id,group_id)
			print >> f, 'Kd %f %f %f'%(new_kd[0],new_kd[1],new_kd[2])
			for subkey in mtl[mtl_id]:
				if subkey == 'Kd': continue
				if subkey == 'map_Kd':
					msg = mtl[mtl_id][subkey]
				else:
					msg = ' '.join([str(v) for v in mtl[mtl_id][subkey]])	
				print >> f, '%s %s'%(subkey, msg)
			print >> f, ''

	print new_mtl_name, 'created'

# change mtl path
# make new mtl name by (mtl_id,group_id)
def write2obj(old_obj_name, new_obj_name, relative_new_mtl_name, obj):
	with open(new_obj_name,'w') as new_f:
		group_id = None
		for line in open(old_obj_name, "r"):
			if line.startswith('mtllib'): 
				line = 'mtllib %s\n'%(relative_new_mtl_name)
			elif line.startswith('g '):
				assert len(line.split())==2, line
				group_id = line.split()[1]
			elif  line.startswith('usemtl ') or line.startswith('usemat ') :
				assert len(line.split())==2, line
				vs = line.split()
				line = '%s %s-%s\n'%(vs[0],vs[1],group_id)
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
		print 'usage: python vox2obj.py vox.npy oldobj.obj output/path/prefix'
		exit(0)
	# input 
	voxpath = sys.argv[1]
	npy = np.load(voxpath)
	shape = npy.shape
	if len(shape)==2 and shape[1]==6:	# points
		vox = points2vox(npy, 64)
	elif len(shape)==4:					# voxels
		vox = npy
	else:
		print 'shape not supported', shape
		exit(0)
	
	# lab -> rgb
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

	# group faces by (mtl_id) and xyz in grid world, {(mtl_id,group_id):{(x,y,z):rgb}}
	mtlkd_dict = {}
	print 'face count:',len(obj.faces)
	for face in obj.faces:
		vertex_ids, normals_ids, tex_ids, mtl_id, group_id = face
		v1 = obj.getVertex(vertex_ids[0])
		v2 = obj.getVertex(vertex_ids[1])
		v3 = obj.getVertex(vertex_ids[2])
		grid_xyz_list = face2grid(v1,v2,v3,origin,scale)
		for x,y,z in grid_xyz_list:
			rgb = vox[x,y,z,0:3]
			key = (mtl_id, group_id)
			# if xyz correpond to an empty vox, 
			if vox[x,y,z,3] == 0:	
				if key not in mtlkd_dict:
					mtlkd_dict[key] = {}
				continue
			# if xyz correpond to a valid vox, 
			if key in mtlkd_dict:
				if (x,y,z) not in mtlkd_dict[key]:
					mtlkd_dict[key][(x,y,z)] = rgb
			else:
				mtlkd_dict[key] = {}
				mtlkd_dict[key][(x,y,z)] = rgb

	# calculate average rgb for each (mtl_id,group_id), {(mtl_id,group_id):rgb}
	# with open('log.txt','w') as f:
	# 	print >> f, mtlkd_dict
	for key in mtlkd_dict:
		if len(mtlkd_dict[key]) == 0:
			mtlkd_dict[key] = np.array([0,0,0])
		else:
			rgbs = np.array(mtlkd_dict[key].values())
			print key, rgbs.shape
			# mtlkd_dict[key] = np.mean(rgbs, axis=0)
			mtlkd_dict[key] = np.median(rgbs, axis=0)


	# compose new mtl file
	dirname, prefix = split(output_path)
	new_mtl_name = join(dirname, prefix+'.mtl')
	write2mtl(new_mtl_name, obj.mtl, mtlkd_dict)

	# compose new obj file
	new_obj_name = join(dirname, prefix+'.obj')
	write2obj(old_obj_name, new_obj_name, prefix+'.mtl', obj)
	















