#python3 setup.py build develop #--no-deps # for building d2
export PYTHONPATH=$PYTHONPATH:`pwd`
# export CUDA_LAUNCH_BLOCKING=1 # for debug
export CUDA_VISIBLE_DEVICES=0

ID=159

# TODO: specify experiment config
## kins
#config_path="configs/KINS-AmodalSeg/aisformer_R_50_FPN_1x_amodal_kins.yaml"
#output_dir="/work/weientai18/aisformer/full_training"
## d2sa
#config_path="configs/D2SA-AmodalSeg/aisformer_R_50_FPN_1x_amodal_d2sa.yaml"

## cocoa
config_path="configs/COCOA_cls-AmodalSeg/aisformer_R_50_FPN_1x_amodal_cocoa_cls.yaml"
output_dir="/work/weientai18/aisformer/full_training_cocoa"
python3 tools/train_net.py --num-gpus 1 --config-file ${config_path} 2>&1 | tee ${output_dir}/train_log_$ID.txt
