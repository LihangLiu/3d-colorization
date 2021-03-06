set -e 

render_obj() {
	python obj2pov.py $2 $3
	cp $povpath $1	# set up rendering settings
	cd $1
	povfile=`basename $povpath`
	povray -H900 -W1200 +A -D +o$4  $povfile
	chmod 775 $4
	cd $py_dir
}

# check arguments
if [ "$#" -lt 2 ]; then
		echo "usage: batch_obj2labpoints.sh path/to/renderroot render_setting.pov"
		exit 1
fi
echo "render under $1 with $2"
povpath=$2

# go over subfolders
py_dir=`pwd`
for subdir in $1/*/ ; do
    echo "$subdir"

    # render original obj first
    python rm_overlapped_meshes.py $subdir/model_normalized.obj
    render_obj $subdir $subdir/model_normalized.obj.rm.obj $subdir/model.inc model_normalized.rm.png 
	
	# render all npy files
	for npyfile in $subdir/*.npy ; do
	 	echo "$npyfile"
	 	new_prefix=`basename $npyfile`
	 	python vox2obj_gsemantic.py $npyfile $subdir/model_normalized.obj.rm.obj $subdir/$new_prefix
	 	render_obj $subdir $subdir/$new_prefix.obj $subdir/model.inc $new_prefix.png &
	done
	wait
done

