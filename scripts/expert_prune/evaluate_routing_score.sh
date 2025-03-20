#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ALLOW_CODE_EVAL="1"

. local_scripts/expert_prune/eval_config.sh

model_id="Moonlight-16B-A3B-Instruct"
torchrun --standalone --nproc_per_node 4 modepd/eval.py \
    --hf_model exp/expert/$model_id-pruned-48 $eval_config \
    --batch_size 8 --trust_remote_code --output_dir exp/eval

torchrun --standalone --nproc_per_node 4 modepd/eval.py \
    --hf_model exp/expert/$model_id-pruned-32 $eval_config \
    --batch_size 8 --trust_remote_code --output_dir exp/eval

model_id="DeepSeek-V2-Lite-Chat"
torchrun --standalone --nproc_per_node 4 modepd/eval.py \
    --hf_model exp/expert/$model_id-pruned-48 $eval_config \
    --batch_size 8 --trust_remote_code --output_dir exp/eval

torchrun --standalone --nproc_per_node 4 modepd/eval.py \
    --hf_model exp/expert/$model_id-pruned-32 $eval_config \
    --batch_size 8 --trust_remote_code --output_dir exp/eval
