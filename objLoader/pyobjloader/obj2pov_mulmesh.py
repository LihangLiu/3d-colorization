# 
# convert vox to a new obj file with the help of original obj file.
# usage: python vox2obj_gsemantic.py vox.npy oldobj.obj output/path
# vox.npy: vox or points

import sys
from os.path import join
import numpy as np
import random
from skimage import io, color
import operator 

from objloader import *

# mesh
def write2pov(pov_path, obj):
	with open(pov_path,'w') as f:
		# write textures
		for mtl_id in obj.mtl:
			Kd = obj.mtl[mtl_id]['Kd']
			Ks = np.mean(obj.mtl[mtl_id]['Ks'])
			Ka = np.mean(obj.mtl[mtl_id]['Ka'])
			illum = np.mean(obj.mtl[mtl_id]['illum'])
			f.write('#declare %s = texture {\n'%(mtl_id))
			f.write('	pigment { color rgb<%f,%f,%f>}\n'%(Kd[0],Kd[1],Kd[2]))
			f.write('	finish  { specular %f ambient %f phong %f}\n'%(Ks,Ka,illum))
			f.write('}\n')
		
		# group faces by group_id into different meshes, faces_dict: {group_id: [face]}
		faces_dict = {}
		for face in obj.faces:
			vertex_ids, normals_ids, tex_ids, mtl_id, group_id = face
			if group_id not in faces_dict:
				faces_dict[group_id] = []
			faces_dict[group_id].append(face)	
		# write meshes 
		for group_id in faces_dict:
			faces = faces_dict[group_id]
			f.write('mesh {\n')
			for vertex_ids, normals_ids, tex_ids, mtl_id, group_id in faces:
				v1 = obj.getVertex(vertex_ids[0])
				v2 = obj.getVertex(vertex_ids[1])
				v3 = obj.getVertex(vertex_ids[2])
				f.write('	triangle {\n')
				f.write('		<%f,%f,%f>,<%f,%f,%f>,<%f,%f,%f>\n'%tuple(v1+v2+v3))
				f.write('		texture { %s }\n'%(mtl_id))
				f.write('		interior_texture { %s }\n'%(mtl_id))
				f.write('	}\n')
			f.write('}\n')

	print 'generated',pov_path



if __name__ == '__main__':
	print sys.argv
	if len(sys.argv)!=3:
		print 'usage: python vox2obj.py obj.obj output.pov'
		exit(0)
	
	
	obj = OBJ(sys.argv[1], swapyz=False)
	pov_path = sys.argv[2]

	print 'face count:',len(obj.faces)
	print 'mtl count:',len(obj.mtl)

	write2pov(pov_path, obj)
	















