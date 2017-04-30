#!/bin/bash
# usage: batch_obj2points.sh N path/to/objroot

# Any subsequent(*) commands which fail will cause the shell script to exit immediately
set -e


obj2points() {
	obj_dir=`dirname $2`
	obj_name=`basename $2`
    # cd to the vox directory
    cd $obj_dir
    #echo -en "\r\033[K $1"
    points_name=$obj_name.$1.points.npy
    if [ -f $points_name ];
    then
            echo -en "\r\033[Kexisting $points_name"
    else
            python $py_dir/obj2points.py $1 $obj_name $points_name
            chmod 755 $points_name
            echo "converted to $obj_dir/$points_name"
    fi
}

# check arguments
if [ "$#" -lt 2 ]; then
        echo "usage: batch_vox2image.sh N path/to/objroot"
        exit 1
fi
echo "lookup *.obj under $2"

obj_paths=`find $2 -name '*.obj'`
IFS=$'\n'
obj_paths=($obj_paths)
echo "Total objs:${#obj_paths[@]}"

py_dir=`pwd`
for (( i=0; i<${#obj_paths[@]}/8; i++ ));
do
	for ((j=0; j<8; j++)); do 
		obj2points $1 ${obj_paths[$i*8+j]} &
	done
	wait
done

echo ""
echo "DONE"


