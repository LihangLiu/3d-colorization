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


# mtl_intid_dict: {mtl_id:int}
def write2pov(pov_path, obj, mtl_intid_dict):
	with open(pov_path,'w') as f:
		f.write('mesh2 {\n')

		# vertex_vectors
		num = len(obj.vertices)
		f.write('   vertex_vectors {\n')
		f.write('      %d,\n'%(num))
		for i,[x,y,z] in enumerate(obj.vertices):
			if i == num-1:
				f.write('      <%f,%f,%f>\n'%(x,y,z));
			else:
				f.write('      <%f,%f,%f>,\n'%(x,y,z));
		f.write('   }\n')

		# normal_vectors
		num = len(obj.normals)
		f.write('   normal_vectors {\n')
		f.write('      %d,\n'%(num))
		for i,[x,y,z] in enumerate(obj.normals):
			if i == num-1:
				f.write('      <%f,%f,%f>\n'%(x,y,z));
			else:
				f.write('      <%f,%f,%f>,\n'%(x,y,z));
		f.write('   }\n')

		# texture
		sorted_x = sorted(mtl_intid_dict.items(), key=operator.itemgetter(1))
		num = len(sorted_x)
		f.write('   texture_list {\n')
		f.write('      %d,\n'%(num))
		for mtl_id,int_id in sorted_x:
			Kd = obj.mtl[mtl_id]['Kd']
			Ks = np.mean(obj.mtl[mtl_id]['Ks'])
			Ka = np.mean(obj.mtl[mtl_id]['Ka'])
			illum = np.mean(obj.mtl[mtl_id]['illum'])
			f.write('      texture {\n')
			f.write('        pigment { color rgb<%f,%f,%f>}\n'%(Kd[0],Kd[1],Kd[2]))
			f.write('        finish  { specular %f ambient %f phong %f}\n'%(Ks,Ka,illum))
			f.write('      }\n')
		f.write('   }\n')

		# face_indices
		num = len(obj.faces)
		f.write('   face_indices {\n')
		f.write('      %d,\n'%(num))
		for i,[vIds,vnIds,tex_ids,mtl_id,group_id] in enumerate(obj.faces):
			int_id = mtl_intid_dict[mtl_id]
			if i == num-1:
				f.write('      <%d,%d,%d>,%d\n'%(vIds[0]-1,vIds[1]-1,vIds[2]-1,int_id));
			else:
				f.write('      <%d,%d,%d>,%d,\n'%(vIds[0]-1,vIds[1]-1,vIds[2]-1,int_id));
		f.write('   }\n')

		# normal_indices
		num = len(obj.faces)
		f.write('   normal_indices {\n')
		f.write('      %d,\n'%(num))
		for i,[vIds,vnIds,tex_ids,mtl_id,group_id] in enumerate(obj.faces):
			if i == num-1:
				f.write('      <%d,%d,%d>\n'%(vnIds[0]-1,vnIds[1]-1,vnIds[2]-1));
			else:
				f.write('      <%d,%d,%d>,\n'%(vnIds[0]-1,vnIds[1]-1,vnIds[2]-1));
		f.write('   }\n')

		f.write('}\n')

	print 'generated'		,pov_path



if __name__ == '__main__':
	print sys.argv
	if len(sys.argv)!=3:
		print 'usage: python vox2obj.py obj.obj output.pov'
		exit(0)
	
	
	obj = OBJ(sys.argv[1], swapyz=False)
	pov_path = sys.argv[2]

	# get all mtl and assign int id, {mtl_id:int}, start from 0
	mtl_intid_dict = {}
	print 'face count:',len(obj.faces)
	print 'mtl count:',len(obj.mtl)
	for i,mtl_id in enumerate(obj.mtl.keys()):
		mtl_intid_dict[mtl_id] = i

	# compose new pov
	write2pov(pov_path, obj, mtl_intid_dict)
	















