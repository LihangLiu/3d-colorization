import sys
from os.path import join
import numpy as np
import random
from skimage import io, color

from objloader import *

# compose from obj class
# remove redundant faces
# texture not considered
def write2obj(new_obj_name, obj):
	# group face by (mtl_id,group_id), mesh_dict: {(mtl_id,group_id):[face]}
	mesh_dict = {}
	for face in obj.faces:
		vertex_ids, normals_ids, tex_ids, mtl_id, group_id = face
		key = (mtl_id,group_id)
		if key not in mesh_dict:
			mesh_dict[key] = []
		mesh_dict[key].append(face)
	# write
	with open(new_obj_name,'w') as f:
		print >> f, '# obj file with redundant faces removed'
		print >> f, 'mtllib model_normalized.mtl'		# tbd: mtl name
		# v
		for x,y,z in obj.vertices:
			print >> f, 'v %f %f %f'%(x,y,z)
		print >> f, ''
		# vn
		for x,y,z in obj.normals:
			print >> f, 'vn %f %f %f'%(x,y,z)
		print >> f, ''
		# texcoords
		for u,v in obj.texcoords:
			print >> f, 'vt %f %f'%(u,v)
		print >> f, ''
		# faces
		for key in mesh_dict:
			mtl_id,group_id = key
			print >> f, 'g '+group_id
			print >> f, 'usemtl '+mtl_id
			for vertex_ids, normals_ids, tex_ids, mtl_id, group_id in mesh_dict[key]:
				msg = 'f %d//%d %d//%d %d//%d'%(vertex_ids[0],normals_ids[0],
												vertex_ids[1],normals_ids[1],
												vertex_ids[2],normals_ids[2])
				print >> f, msg
		print >> f, ''

	print 'generated', new_obj_name

def encode_3v(vid0,vid1,vid2):
	res = [vid0,vid1,vid2]
	res.sort()
	return '%d-%d-%d'%(res[0],res[1],res[2])

if __name__ == '__main__':
	print sys.argv
	if len(sys.argv)!=2:
		print 'usage: python vox2obj.py oldobj.obj'
		exit(0)
	# input 
	old_obj_path = sys.argv[1]
	obj = OBJ(old_obj_path, swapyz=False)

	# group faces by mtl_id, mtl_dict: {mtl_id : faceset}, 
	# faceset: (v1-v2-v3), ascending
	mtl_dict = {}
	for vertex_ids, normals_ids, tex_ids, mtl_id, group_id in obj.faces:
		vid0,vid1,vid2 = vertex_ids
		code = encode_3v(vid0,vid1,vid2)
		if mtl_id not in mtl_dict:
			mtl_dict[mtl_id] = set()
		mtl_dict[mtl_id].add(code)

	count = 0
	for key in mtl_dict:
		count += len(mtl_dict[key])
	print count

	# remove redundant faces, keep that is less
	new_faces = []
	print 'face count:',len(obj.faces)
	for face in obj.faces:
		vertex_ids, normals_ids, tex_ids, mtl_id, group_id = face
		vid0,vid1,vid2 = vertex_ids
		code = encode_3v(vid0,vid1,vid2)
		if_remove = False
		num = len(mtl_dict[mtl_id])
		for c_mtl_id in mtl_dict:
			if len(mtl_dict[c_mtl_id]) < num:
				if code in mtl_dict[c_mtl_id]:
					if_remove = True
					break
		if not if_remove:
			new_faces.append(face)
	obj.faces = new_faces
	print 'new face count:',len(obj.faces)

	new_obj_name = old_obj_path+'.rm.obj'
	write2obj(new_obj_name, obj)



















