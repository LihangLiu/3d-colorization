import glob
import binvox_downsample as binvox
#files = glob.glob('/scratch/cluster/yzp12/dataset/ShapeNetCore.v2/02958343/*/models/*solid.binvox')
files = glob.glob('/scratch/cluster/yzp12/dataset/ShapeNetCore.v2/03001627/*/models/*solid.binvox')
for i, file_path in enumerate(files):
    file_path_ds = file_path[:-7] + "_32.binvox"
    data = binvox.read_binvox(file_path)
    data.dims = [32,32,32]
    data.data = data.data[::4,::4,::4]
    binvox.save_binvox(data, file_path_ds)
    print i
