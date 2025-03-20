#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$PWD:$PYTHONPATH

. local_scripts/expert_prune/data_config.sh

model_id="DeepSeek-V2-Lite-Chat"

python modepd/prune.py --model_name_or_path $model_id $data_config \
    --expert_prune --expert_prune_metric routing_score --preserve_n_experts 48 \
    --compressed_model_save_path exp/expert/$model_id-pruned-48