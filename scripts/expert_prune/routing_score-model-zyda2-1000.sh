#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$PWD:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com

. local_scripts/expert_prune/data_config.sh

model_id="OLMoE-1B-7B-0125"

calib_size=1000
subdir=zyda2-$calib_size

# python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
#     --expert_prune --expert_prune_metric routing_score --preserve_n_experts 48 --expert_ranking_scope model \
#     --compressed_model_save_path exp/$subdir/routing_score_model/$model_id-pruned-48 --max_steps $calib_size

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --expert_prune --expert_prune_metric routing_score --preserve_n_experts 32 --expert_ranking_scope model \
    --compressed_model_save_path exp/$subdir/routing_score_model/$model_id-pruned-32 --max_steps $calib_size

model_id="Moonlight-16B-A3B"

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --expert_prune --expert_prune_metric routing_score --preserve_n_experts 48 --expert_ranking_scope model \
    --compressed_model_save_path exp/$subdir/routing_score_model/$model_id-pruned-48 --max_steps $calib_size

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --expert_prune --expert_prune_metric routing_score --preserve_n_experts 32 --expert_ranking_scope model \
    --compressed_model_save_path exp/$subdir/routing_score_model/$model_id-pruned-32 --max_steps $calib_size

model_id="DeepSeek-V2-Lite"

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --expert_prune --expert_prune_metric routing_score --preserve_n_experts 48 --expert_ranking_scope model \
    --compressed_model_save_path exp/$subdir/routing_score_model/$model_id-pruned-48 --max_steps $calib_size

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/models/$model_id $data_config \
    --expert_prune --expert_prune_metric routing_score --preserve_n_experts 32 --expert_ranking_scope model \
    --compressed_model_save_path exp/$subdir/routing_score_model/$model_id-pruned-32 --max_steps $calib_size
