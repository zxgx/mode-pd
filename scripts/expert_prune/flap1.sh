#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PWD:$PYTHONPATH

. local_scripts/expert_prune/data_config.sh

model_id="Moonlight-16B-A3B-Instruct"

python modepd/prune.py --model_name_or_path $model_id $data_config \
    --weight_prune --weight_prune_metric flap --preserve_channels_in_percent 0.50 \
    --compressed_model_save_path exp/expert/$model_id-flap-pruned-50
