#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com

sft_cache_path=/mnt/videodata/zhgeng/datasets/preprocessed-4k-tulu-3-sft-mixture

# distill
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT modepd/sft.py \
    --model_name_or_path exp/finetune/OLMoE-1B-3B-step2560_1e-4 \
    --distillation --teacher_model_name_or_path /mnt/videodata/zhgeng/models/OLMoE-1B-7B-0125 \
    --sft_cache_path $sft_cache_path --learning_rate 5e-5 --disable_batch_aggregation \
    --lr_scheduler_type "cosine" --gradient_accumulation_steps 2 --num_warmup_steps 500 \
    --with_tracking --output_dir exp/distill/OLMoE-1B-3B --evaluate_every 500
