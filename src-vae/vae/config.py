train_dataset_path = "../../data/ShapeNetCore.v2/datav2/data/lists/lab_npy_list.02958343.64.points.txt.train"
test_dataset_path = "../../data/ShapeNetCore.v2/datav2/data/lists/lab_npy_list.02958343.64.points.txt.test"

version = 3009

ITER_MIN = 1000
ITER_MAX = 1030
sample_interval = 100
save_interval = 100
pca_path = "../../data/ShapeNetCore.v2/data/pca_vectors_02958343.npy"
# preload_model = ""
preload_model = "../../outputs/params/params%d_%d.ckpt"%(version,ITER_MIN)

loss_csv = "../../outputs/losses/loss_%d.csv"%(version)
vox_prefix = "../../outputs/voxels/epoch%d_"%(version)
param_prefix = "../../outputs/params/params%d_"%(version)
log_txt = "../../outputs/log/log_%d.txt"%(version)
