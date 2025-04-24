#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PWD:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com

. local_scripts/expert_prune/data_config.sh

model_id="OLMoE-1B-7B-0125"

calib_size=500
subdir=zyda2-$calib_size

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --weight_prune --weight_prune_metric sparsegpt --sparsegpt_prunen 1 --sparsegpt_prunem 4 \
    --compressed_model_save_path exp/$subdir/sparsegpt/$model_id-sparsegpt-pruned-75 --max_steps $calib_size

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --weight_prune --weight_prune_metric sparsegpt --sparsegpt_prunen 2 --sparsegpt_prunem 4 \
    --compressed_model_save_path exp/$subdir/sparsegpt/$model_id-sparsegpt-pruned-50 --max_steps $calib_size

model_id="Moonlight-16B-A3B"

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --weight_prune --weight_prune_metric sparsegpt --sparsegpt_prunen 1 --sparsegpt_prunem 4 \
    --compressed_model_save_path exp/$subdir/sparsegpt/$model_id-sparsegpt-pruned-75 --max_steps $calib_size

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --weight_prune --weight_prune_metric sparsegpt --sparsegpt_prunen 2 --sparsegpt_prunem 4 \
    --compressed_model_save_path exp/$subdir/sparsegpt/$model_id-sparsegpt-pruned-50 --max_steps $calib_size

model_id="DeepSeek-V2-Lite"

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --weight_prune --weight_prune_metric sparsegpt --sparsegpt_prunen 1 --sparsegpt_prunem 4 \
    --compressed_model_save_path exp/$subdir/sparsegpt/$model_id-sparsegpt-pruned-75 --max_steps $calib_size

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --weight_prune --weight_prune_metric sparsegpt --sparsegpt_prunen 2 --sparsegpt_prunem 4 \
    --compressed_model_save_path exp/$subdir/sparsegpt/$model_id-sparsegpt-pruned-50 --max_steps $calib_size
