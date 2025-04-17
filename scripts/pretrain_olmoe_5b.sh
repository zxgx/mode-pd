#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com

# the continued pretraining reported file system error at step 4608 
# we can run the follow cmd with --skip_train to convert the model checkpoint to intermediate model
# torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK --nproc_per_node=$GPUS_PER_NODE \
#     --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT modepd/train.py \
#     --model_name_or_path exp/layer/Moonlight-16B-A3B-Instruct-layer-pruned-19 \
#     --dataset_name_or_path /mnt/videodata/zhgeng/Zyda-2/sample/100BT --streaming_dataset \
#     --validation_dataset_name_or_path /mnt/videodata/zhgeng/datasets/wikitext \
#     --output_dir exp/finetune/bak --skip_train --resume_from_checkpoint exp/finetune/bak

# resume training from the checkpoint at step 4608
# one stop - 100B
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT modepd/train.py \
    --model_name_or_path exp/mone_rs_out_fusion_layer/OLMoE-1B-7B-0125-mone-pruned-48 \
    --dataset_name_or_path /mnt/videodata/zhgeng/OLMoE-mix-0924 --streaming_dataset \
    --validation_dataset_name_or_path /mnt/videodata/zhgeng/datasets/wikitext \
    --per_device_train_batch_size 1 --max_train_steps 25600 --num_warmup_steps 512 --gradient_accumulation_steps 32 \
    --with_tracking --output_dir exp/finetune/OLMoE-1B-5B-step25600_resume \
    --checkpointing_steps 512 --evaluate_every 512 --resume_from_checkpoint exp/finetune/OLMoE-1B-5B-step25600
