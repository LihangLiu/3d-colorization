dataset_path = "../../data/npy_list.64.points.txt"
#dataset_path = "../../data/npy_list.64.txt"

version = 1002

ITER_MIN = 1
ITER_MAX = 501
save_interval = 10
preload_model = ""
# preload_model = "../../outputs/params/params%d_%d.ckpt"%(version,ITER_MIN)

loss_csv = "../../outputs/losses/loss_%d.csv"%(version)
vox_prefix = "../../outputs/voxels/epoch%d_"%(version)
param_prefix = "../../outputs/params/params%d_"%(version)
log_txt = "../../outputs/log/log_%d.txt"%(version)

