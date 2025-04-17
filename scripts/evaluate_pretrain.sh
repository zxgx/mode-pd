#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ALLOW_CODE_EVAL="1"

GPUS_PER_NODE=8

torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
    --hf_model exp/finetune/OLMoE-1B-5B-step25600_resume \
    --batch_size 8 --trust_remote_code --output_dir exp/finetune
