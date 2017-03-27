#!/bin/bash
# multithread version
# usage: batch_vox2image.sh path/to/npy/folder
#

# Any subsequent(*) commands which fail will cause the shell script to exit immediately
set -e

vox2image() {
	vox_dir=`dirname $1`
	vox_name=`basename $1`
    # cd to the vox directory
    cd $vox_dir
    #echo -en "\r\033[K $1"
    if [ -f "$vox_name.jpg" ];
    then
            echo -en "\r\033[Kexisting $1"
    else
            python $py_dir/vox2image.py $vox_name $vox_name.jpg
            chmod 755 $vox_name.jpg
            echo "converted to $vox_dir/$vox_name.jpg"
    fi
}

#py_dir=`dirname $(pwd)/$0`
#echo $py_dir
#exit 1

# check arguments
if [ "$#" -lt 1 ]; then
        echo "usage: batch_vox2image.sh path/to/npy/folder [name_rule]"
        exit 1
fi
rule="*.npy"
if [ "$#" -eq 2 ]; then
        rule=$2
fi
echo "searching for $rule under $1"

vox_paths=`find $1 -name "$rule"`
IFS=$'\n'
vox_paths=($vox_paths)
echo "Total voxels:${#vox_paths[@]}"

py_dir=`pwd`
for (( i=0; i<${#vox_paths[@]}/8; i++ ));
do
	for ((j=0; j<8; j++)); do 
		vox2image ${vox_paths[$i*8+j]} &
	done
	wait
done

echo ""
echo "DONE"
