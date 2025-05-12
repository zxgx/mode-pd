#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ALLOW_CODE_EVAL="1"
GPUS_PER_NODE=8

# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model /mnt/videodata/zhgeng/models/OLMoE-1B-7B-0125 \
#     --batch_size 8 --trust_remote_code --output_dir exp/pretrain/baseline

# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model /mnt/videodata/zhgeng/models/Qwen2.5-0.5B \
#     --batch_size 8 --trust_remote_code --output_dir exp/pretrain/baseline

# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model /mnt/videodata/zhgeng/models/Qwen2.5-1.5B \
#     --batch_size 8 --trust_remote_code --output_dir exp/pretrain/baseline

# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model /mnt/videodata/zhgeng/models/gemma-3-1b-pt \
#     --batch_size 8 --trust_remote_code --output_dir exp/pretrain/baseline
