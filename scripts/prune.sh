#!/bin/bash

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=$PWD:$PYTHONPATH

python modepd/prune.py --model_name_or_path /mnt/workspace/zhgeng/DeepSeek-V2-Lite-Chat --expert_prune --preserve_n_experts 48
