# Install
	python setup.py install --user

# Usage
	python obj2vox.py path/to/objfile.obj [path/to/savednpy.npy]

# Batch conversion
  convert all .obj to .npy(32,32,32,4) under a certain directory
  
	run_batch.sh /path/to/data/root

# Visualize .npy file
  rgb \in (0,1). All non-zero alpha is considered to be 1.

	python visvox.py path/to/npyfile.npy
