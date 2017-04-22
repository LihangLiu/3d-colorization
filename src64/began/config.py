train_dataset_path = "../../data/npy_list.02958343.64.points.txt.train"

version = 1101

ITER_MIN = 450
ITER_MAX = 1001
save_interval = 50
#preload_model = ""
preload_model = "../../outputs/params/params%d_%d.ckpt"%(version,ITER_MIN)

loss_csv = "../../outputs/losses/loss_%d.csv"%(version)
vox_prefix = "../../outputs/voxels/epoch%d_"%(version)
param_prefix = "../../outputs/params/params%d_"%(version)
log_txt = "../../outputs/log/log_%d.txt"%(version)

