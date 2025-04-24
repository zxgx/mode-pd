#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com

. local_scripts/expert_prune/data_config.sh

model_id="OLMoE-1B-7B-0125"

python modepd/prune.py --model_name_or_path /mnt/videodata/zhgeng/models/$model_id $data_config \
    --expert_prune --preserve_n_experts 48 --expert_ranking_scope layer \
    --expert_prune_metric mone --mone_ranking_metric output_fluctuation \
    --compressed_model_save_path exp/mone_out/$model_id-mone-pruned-48

python modepd/prune.py --model_name_or_path /mnt/videodata/zhgeng/models/$model_id $data_config \
    --expert_prune --preserve_n_experts 32 --expert_ranking_scope layer \
    --expert_prune_metric mone --mone_ranking_metric output_fluctuation \
    --compressed_model_save_path exp/mone_out/$model_id-mone-pruned-32

model_id="Moonlight-16B-A3B"

python modepd/prune.py --model_name_or_path /mnt/videodata/zhgeng/models/$model_id $data_config \
    --expert_prune --preserve_n_experts 48 --expert_ranking_scope layer \
    --expert_prune_metric mone --mone_ranking_metric output_fluctuation \
    --compressed_model_save_path exp/mone_out/$model_id-mone-pruned-48

python modepd/prune.py --model_name_or_path /mnt/videodata/zhgeng/models/$model_id $data_config \
    --expert_prune --preserve_n_experts 32 --expert_ranking_scope layer \
    --expert_prune_metric mone --mone_ranking_metric output_fluctuation \
    --compressed_model_save_path exp/mone_out/$model_id-mone-pruned-32

model_id="DeepSeek-V2-Lite"

python modepd/prune.py --model_name_or_path /mnt/videodata/zhgeng/models/$model_id $data_config \
    --expert_prune --preserve_n_experts 48 --expert_ranking_scope layer \
    --expert_prune_metric mone --mone_ranking_metric output_fluctuation \
    --compressed_model_save_path exp/mone_out/$model_id-mone-pruned-48

python modepd/prune.py --model_name_or_path /mnt/videodata/zhgeng/models/$model_id $data_config \
    --expert_prune --preserve_n_experts 32 --expert_ranking_scope layer \
    --expert_prune_metric mone --mone_ranking_metric output_fluctuation \
    --compressed_model_save_path exp/mone_out/$model_id-mone-pruned-32
