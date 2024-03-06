#python3 setup.py build develop #--no-deps # for building d2
export PYTHONPATH=$PYTHONPATH:`pwd`
# export CUDA_LAUNCH_BLOCKING=1 # for debug
export CUDA_VISIBLE_DEVICES=0

ID=159


# trained model path and config
output_dir="/work/weientai18/"
#test_ckpt="model_0119999_best.pth"
#python3 tools/train_net_aissam.py --num-gpus 1 \
#        --config-file ${output_dir}/config.yaml \
#        --eval-only MODEL.WEIGHTS ${output_dir}/${test_ckpt} 2>&1 | tee ${output_dir}/sam_log.txt
python3 tools/test_net_aissam.py tee ${output_dir}/sam_test_log.txt
