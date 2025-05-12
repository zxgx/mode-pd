#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com

# one stop - 10B
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT modepd/train.py \
    --model_name_or_path exp/mone_rs_out_fusion_layer/OLMoE-1B-7B-0125-mone-pruned-32 \
    --dataset_name_or_path /mnt/videodata/zhgeng/OLMoE-mix-0924 --streaming_dataset \
    --validation_dataset_name_or_path /mnt/videodata/zhgeng/datasets/wikitext \
    --per_device_train_batch_size 1 --max_train_steps 2560 --num_warmup_steps 256 --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine_with_min_lr --learning_rate 1e-4 --min_lr 1e-5 \
    --with_tracking --output_dir exp/finetune/OLMoE-1B-3B-step2560_1e-4 \
    --checkpointing_steps 512 --evaluate_every 256
