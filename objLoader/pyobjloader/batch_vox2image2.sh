#!/bin/bash
# multithread version
#
#

# Any subsequent(*) commands which fail will cause the shell script to exit immediately
set -e

vox2image() {
	vox_dir=`dirname $1`
	vox_name=`basename $1`
    # cd to the vox directory
    cd $vox_dir
    echo $1
    if [ -f "$vox_name.jpg" ];
    then
            echo "existing."
    else
            python $py_dir/vox2image.py $vox_name $vox_name.jpg
            chmod 755 $vox_name.jpg
            echo "converted to $vox_name.jpg"
    fi
}

py_dir=`pwd`
vox_paths=`find $1 -name '*.npy'`
IFS=$'\n'
vox_paths=($vox_paths)
echo ${#vox_paths[@]}

for (( i=0; i<${#vox_paths[@]}/8; i++ ));
do
	for ((j=0; j<8; j++)); do 
		vox2image ${vox_paths[$i*8+j]} &
	done
	wait
done



