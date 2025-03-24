#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PWD:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com

. local_scripts/expert_prune/data_config.sh

model_id=<model path> #"Moonlight-16B-A3B-Instruct"

python modepd/prune.py --model_name_or_path $model_id $data_config \
    --expert_prune --expert_prune_metric mc_smoe --preserve_n_experts 48 \
    --compressed_model_save_path exp/mc_smoe/$model_id-mc_smoe-pruned-48

python modepd/prune.py --model_name_or_path $model_id $data_config \
    --expert_prune --expert_prune_metric mc_smoe --preserve_n_experts 32 \
    --compressed_model_save_path exp/mc_smoe/$model_id-mc_smoe-pruned-32

model_id=<model path> #"DeepSeek-V2-Lite-Chat"

python modepd/prune.py --model_name_or_path $model_id $data_config \
    --expert_prune --expert_prune_metric mc_smoe --preserve_n_experts 48 \
    --compressed_model_save_path exp/mc_smoe/$model_id-mc_smoe-pruned-48
    
python modepd/prune.py --model_name_or_path $model_id $data_config \
    --expert_prune --expert_prune_metric mc_smoe --preserve_n_experts 32 \
    --compressed_model_save_path exp/mc_smoe/$model_id-mc_smoe-pruned-32
