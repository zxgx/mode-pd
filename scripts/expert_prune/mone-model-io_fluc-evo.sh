#!/bin/bash

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=$PWD:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com

. local_scripts/expert_prune/data_config.sh

model_id=<model path> #"Moonlight-16B-A3B-Instruct"

python modepd/prune.py --model_name_or_path $model_id $data_config \
    --expert_prune --preserve_n_experts 48 --expert_ranking_scope model \
    --expert_prune_metric mone --mone_ranking_metric io_fluctuation --enable_novice_evolving \
    --compressed_model_save_path exp/mone_io_fluctuation_evolve_model/$model_id-mone-pruned-48

python modepd/prune.py --model_name_or_path $model_id $data_config \
    --expert_prune --preserve_n_experts 32 --expert_ranking_scope model \
    --expert_prune_metric mone --mone_ranking_metric io_fluctuation --enable_novice_evolving \
    --compressed_model_save_path exp/mone_io_fluctuation_evolve_model/$model_id-mone-pruned-32

model_id=<model path> #"DeepSeek-V2-Lite-Chat"

python modepd/prune.py --model_name_or_path $model_id $data_config \
    --expert_prune --preserve_n_experts 48 --expert_ranking_scope model \
    --expert_prune_metric mone --mone_ranking_metric io_fluctuation --enable_novice_evolving \
    --compressed_model_save_path exp/mone_io_fluctuation_evolve_model/$model_id-mone-pruned-48

python modepd/prune.py --model_name_or_path $model_id $data_config \
    --expert_prune --preserve_n_experts 32 --expert_ranking_scope model \
    --expert_prune_metric mone --mone_ranking_metric io_fluctuation --enable_novice_evolving \
    --compressed_model_save_path exp/mone_io_fluctuation_evolve_model/$model_id-mone-pruned-32
