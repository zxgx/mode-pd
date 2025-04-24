#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com

model_path=exp/flap/OLMoE-1B-7B-0125-flap-pruned-75
output_path=exp/pretrain/flap

GLOBAL_TOKEN_SIZE=$((4*1024*1024))
BLOCK_SIZE=4096  # Context length
dp_size=$GPUS_PER_NODE*$WORLD_SIZE
GRAD_ACCUM_STEPS=$((GLOBAL_TOKEN_SIZE/BLOCK_SIZE/dp_size))

# one stop - 2B
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT modepd/train.py \
    --model_name_or_path $model_path \
    --dataset_name_or_path /mnt/videodata/zhgeng/OLMoE-mix-0924 --streaming_dataset \
    --validation_dataset_name_or_path /mnt/videodata/zhgeng/datasets/wikitext \
    --per_device_train_batch_size 1 --max_train_steps 512 --num_warmup_steps 50 --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --lr_scheduler_type cosine_with_min_lr --learning_rate 5e-5 --min_lr 5e-6 \
    --with_tracking --output_dir $output_path --evaluate_every 50
