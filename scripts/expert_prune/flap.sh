#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com

. local_scripts/expert_prune/data_config.sh

model_id=<model path> #"Moonlight-16B-A3B-Instruct"

python modepd/prune.py --model_name_or_path $model_id $data_config \
    --weight_prune --weight_prune_metric flap --preserve_channels_in_percent 0.75 \
    --compressed_model_save_path exp/flap/$model_id-flap-pruned-75

python modepd/prune.py --model_name_or_path $model_id $data_config \
    --weight_prune --weight_prune_metric flap --preserve_channels_in_percent 0.50 \
    --compressed_model_save_path exp/flap/$model_id-flap-pruned-50

model_id=<model path> #"DeepSeek-V2-Lite-Chat"

python modepd/prune.py --model_name_or_path $model_id $data_config \
    --weight_prune --weight_prune_metric flap --preserve_channels_in_percent 0.75 \
    --compressed_model_save_path exp/flap/$model_id-flap-pruned-75

python modepd/prune.py --model_name_or_path $model_id $data_config \
    --weight_prune --weight_prune_metric flap --preserve_channels_in_percent 0.50 \
    --compressed_model_save_path exp/flap/$model_id-flap-pruned-50
