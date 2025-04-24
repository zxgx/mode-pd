#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH=$PWD:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com

. local_scripts/expert_prune/data_config_c4.sh

model_id="OLMoE-1B-7B-0125"

calib_size=1000
subdir=c4-$calib_size

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --layer_prune --layer_prune_metric angular --drop_n_layers 4 \
    --compressed_model_save_path exp/$subdir/angular/$model_id-angular-pruned-75 --max_steps $calib_size

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --layer_prune --layer_prune_metric angular --drop_n_layers 8 \
    --compressed_model_save_path exp/$subdir/angular/$model_id-angular-pruned-50 --max_steps $calib_size

model_id="Moonlight-16B-A3B"

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --layer_prune --layer_prune_metric angular --drop_n_layers 6 \
    --compressed_model_save_path exp/$subdir/angular/$model_id-angular-pruned-75 --max_steps $calib_size

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --layer_prune --layer_prune_metric angular --drop_n_layers 13 \
    --compressed_model_save_path exp/$subdir/angular/$model_id-angular-pruned-50 --max_steps $calib_size

model_id="DeepSeek-V2-Lite"

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --layer_prune --layer_prune_metric angular --drop_n_layers 6 \
    --compressed_model_save_path exp/$subdir/angular/$model_id-angular-pruned-75 --max_steps $calib_size

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --layer_prune --layer_prune_metric angular --drop_n_layers 13 \
    --compressed_model_save_path exp/$subdir/angular/$model_id-angular-pruned-50 --max_steps $calib_size
