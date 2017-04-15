train_dataset_path = "../../data/npy_list.02958343.64.points.txt.train"
test_dataset_path = "../../data/npy_list.02958343.64.points.txt.test"

version = 3002

ITER_MIN = 1
ITER_MAX = 501
save_interval = 10
pca_path = "pca_vectors.npy"
preload_model = ""
# preload_model = "../../outputs/params/params%d_%d.ckpt"%(version,ITER_MIN)

loss_csv = "../../outputs/losses/loss_%d.csv"%(version)
vox_prefix = "../../outputs/voxels/epoch%d_"%(version)
param_prefix = "../../outputs/params/params%d_"%(version)
log_txt = "../../outputs/log/log_%d.txt"%(version)
