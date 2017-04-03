#!/bin/bash
# usage: batch_obj2points.sh path/to/objroot

# Any subsequent(*) commands which fail will cause the shell script to exit immediately
set -e

py_dir=`pwd`
obj_paths=`find $1 -name '*.obj'`
for obj_path in $obj_paths; do
	obj_dir=`dirname $obj_path`
	# cd to the obj directory
	cd $obj_dir
	obj_name=`basename $obj_path`
	pwd
#	if [ -f "$obj_name.npy" ];
#	then
#		echo "exist."
#	else
#		echo "converting ..."
#		python $py_dir/obj2vox.py $obj_name $obj_name.npy
#	fi
	python $py_dir/obj2points.py 32  $obj_name $obj_name.32.points.npy
	chmod 755 $obj_name.npy
done

