train_dataset_path = "../../data/npy_list.all.64.points.txt.train"
test_dataset_path = "../../data/npy_list.all.64.points.txt.test"

version = 2003

ITER_MIN = 110
ITER_MAX = 201
save_interval = 10
preload_mode = "../../outputs/params/params2003_110.ckpt"

loss_csv = "../../outputs/losses/loss_%d.csv"%(version)
vox_prefix = "../../outputs/voxels/epoch%d_"%(version)
param_prefix = "../../outputs/params/params%d_"%(version)
log_txt = "../../outputs/log/log_%d.txt"%(version)


