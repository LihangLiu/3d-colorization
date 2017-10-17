# 1. obj -> vox
  c++ implementation and python interface. Textures not considered.
  
## Install
	python setup.py install --user

## Usage
	cd parent/path/of/objfile/
	python path/to/obj2vox.py objfile.obj savednpy.npy

## Batch conversion
  convert all .obj of a given directory to .npy(32,32,32,4) and place in the same directory
  
	batch_obj2vox.sh /path/to/objdata/root

# 2. Visualize vox file
  alpha channel: non-0 as 1. 
  
  rgb channels: assume to between (0,1), otherwise clipped to be (0,1).

	python visvox.py path/to/npyfile.npy
	
# 3. vox -> .jpg 
  alpha channel: non-0 as 1. 
  
  rgb channels: assume to between (0,1), otherwise clipped to be (0,1).
  
## Usage
	python vox2image.py path/to/npyfile.npy path/to/savedjpg.jpg
	
## Batch conversion
  convert all .npy of a given directory to .jpg and place in the same directory. Multithreading version.
  
	batch_vox2image.sh /path/to/npydata/root
