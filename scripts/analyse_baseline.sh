#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ALLOW_CODE_EVAL="1"
GPUS_PER_NODE=1

for model_id in OLMoE-1B-7B-0125 Moonlight-16B-A3B DeepSeek-V2-Lite
torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/analyse.py \
    --hf_model /mnt/videodata/zhgeng/models/$model_id \
    --batch_size 8 --trust_remote_code --output_dir exp/pretrain/baseline
