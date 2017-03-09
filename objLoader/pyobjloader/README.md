# Install
	python setup.py install --user

# Usage
	python obj2vox.py path/to/objfile.obj [path/to/savednpy.npy]
	
# Visualize .npy file
	python visvox.py path/to/npyfile.npy
# Batch conversion
  convert all .obj to .npy(32,32,32,4) under a certain directory
  
	run_batch.sh /path/to/data/root

