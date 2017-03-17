#!/bin/bash

# Any subsequent(*) commands which fail will cause the shell script to exit immediately
set -e

py_dir=`pwd`
vox_paths=`find $1 -name '*.npy'`
for vox_path in $vox_paths; do
	vox_dir=`dirname $vox_path`
	# cd to the vox directory
	cd $vox_dir
	vox_name=`basename $vox_path`
	echo $vox_path
	if [ -f "$vox_name.jpg" ];
	then
		echo "exist."
	else
		echo "converting ..."
		python $py_dir/vox2image.py $vox_name $vox_name.jpg
		chmod 755 $vox_name.jpg
	fi
#	python $py_dir/vox2image.py $vox_name $vox_name.jpg
#	chmod 755 $vox_name.jpg
done

