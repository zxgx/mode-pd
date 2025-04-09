#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PWD:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com

. local_scripts/expert_prune/data_config_c4.sh

model_id="OLMoE-1B-7B-0125"

calib_size=100
subdir=c4-$calib_size

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --expert_prune --expert_prune_metric mc_smoe --preserve_n_experts 48 \
    --compressed_model_save_path exp/$subdir/mc_smoe/$model_id-mc_smoe-pruned-48 --max_steps $calib_size

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --expert_prune --expert_prune_metric mc_smoe --preserve_n_experts 32 \
    --compressed_model_save_path exp/$subdir/mc_smoe/$model_id-mc_smoe-pruned-32 --max_steps $calib_size

model_id="Moonlight-16B-A3B"

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --expert_prune --expert_prune_metric mc_smoe --preserve_n_experts 48 \
    --compressed_model_save_path exp/$subdir/mc_smoe/$model_id-mc_smoe-pruned-48 --max_steps $calib_size

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --expert_prune --expert_prune_metric mc_smoe --preserve_n_experts 32 \
    --compressed_model_save_path exp/$subdir/mc_smoe/$model_id-mc_smoe-pruned-32 --max_steps $calib_size

model_id="DeepSeek-V2-Lite"

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --expert_prune --expert_prune_metric mc_smoe --preserve_n_experts 48 \
    --compressed_model_save_path exp/$subdir/mc_smoe/$model_id-mc_smoe-pruned-48 --max_steps $calib_size
    
python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --expert_prune --expert_prune_metric mc_smoe --preserve_n_experts 32 \
    --compressed_model_save_path exp/$subdir/mc_smoe/$model_id-mc_smoe-pruned-32 --max_steps $calib_size
