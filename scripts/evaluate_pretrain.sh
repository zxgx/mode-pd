#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ALLOW_CODE_EVAL="1"

GPUS_PER_NODE=8

. local_scripts/expert_prune/eval_config.sh

# for model_path in angular flap mc_smoe routing_score_model mone; do
for model_path in angular mc_smoe routing_score_model mone; do
    torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py $eval_config \
        --hf_model exp/pretrain/$model_path $eval_config \
        --batch_size 8 --trust_remote_code --output_dir exp/pretrain
done
