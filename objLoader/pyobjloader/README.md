# 1. .obj -> .npy
  c++ implementation and python interface
  
## Install
	python setup.py install --user

## Usage
	python obj2vox.py path/to/objfile.obj path/to/savednpy.npy

## Batch conversion
  convert all .obj of a given directory to .npy(32,32,32,4) and place in the same directory
  
	batch_obj2vox.sh /path/to/objdata/root

# 2. Visualize .npy file
  alpha channel: non-0 as 1. 
  
  rgb channels: assume to between (0,1), otherwise clipped to be (0,1).

	python visvox.py path/to/npyfile.npy
	
# 3. .npy -> .jpg 
  alpha channel: non-0 as 1. 
  
  rgb channels: assume to between (0,1), otherwise clipped to be (0,1).
  
## Usage
	python vox2image.py path/to/npyfile.npy path/to/savedjpg.jpg
	
## Batch conversion
  convert all .npy of a given directory to .jpg and place in the same directory. Multithreading version.
  
	batch_vox2image.sh /path/to/npydata/root
