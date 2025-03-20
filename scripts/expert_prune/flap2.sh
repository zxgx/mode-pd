#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$PWD:$PYTHONPATH

. local_scripts/expert_prune/data_config.sh

model_id="DeepSeek-V2-Lite-Chat"

python modepd/prune.py --model_name_or_path $model_id $data_config \
    --weight_prune --weight_prune_metric flap --preserve_channels_in_percent 0.75 \
    --compressed_model_save_path exp/expert/$model_id-flap-pruned-75
