#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH

GPUS_PER_NODE=8

for model_path in OLMoE-1B-7B-0125 Moonlight-16B-A3B DeepSeek-V2-Lite; do
    torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/train.py \
        --model_name_or_path /mnt/videodata/zhgeng/models/${model_path} \
        --dataset_name_or_path /mnt/videodata/zhgeng/OLMoE-mix-0924 --streaming_dataset \
        --validation_dataset_name_or_path /mnt/videodata/zhgeng/datasets/wikitext \
        --per_device_train_batch_size 1 --skip_train --zero_stage 1 --evaluate_dir exp/baseline
done
