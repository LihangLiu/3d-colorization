# 1. obj -> vox
  python implementation. Textures considered.

	python obj2vox.py path/to/objfile.obj path/to/savednpy.npy

# 2. obj -> pov

	python vox2obj.py obj.obj output.pov
  
	
# 3. remove duplicated triangles in obj
  generate new obj file with path oldobj.obj.rm.obj, while mtl file remains the same
  
  python rm_overlapped_meshes.py oldobj.obj
  
  
# 4. vox -> obj
  generate new obj file with path output_path/model_from_vox.obj and new mtl file with path output_path/model_from_vox.mtl

	python vox2obj.py vox.npy oldobj.obj output/path
