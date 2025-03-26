#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com

. local_scripts/expert_prune/data_config.sh

model_id="OLMoE-1B-7B-0125"

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --weight_prune --weight_prune_metric flap --preserve_channels_in_percent 0.75 \
    --compressed_model_save_path exp/flap/$model_id-flap-pruned-75

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --weight_prune --weight_prune_metric flap --preserve_channels_in_percent 0.50 \
    --compressed_model_save_path exp/flap/$model_id-flap-pruned-50

model_id="Moonlight-16B-A3B"

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --weight_prune --weight_prune_metric flap --preserve_channels_in_percent 0.75 \
    --compressed_model_save_path exp/flap/$model_id-flap-pruned-75

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --weight_prune --weight_prune_metric flap --preserve_channels_in_percent 0.50 \
    --compressed_model_save_path exp/flap/$model_id-flap-pruned-50

model_id="DeepSeek-V2-Lite"

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --weight_prune --weight_prune_metric flap --preserve_channels_in_percent 0.75 \
    --compressed_model_save_path exp/flap/$model_id-flap-pruned-75

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --weight_prune --weight_prune_metric flap --preserve_channels_in_percent 0.50 \
    --compressed_model_save_path exp/flap/$model_id-flap-pruned-50
