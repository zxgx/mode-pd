#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

python modepd/prune.py --model_name_or_path deepseek-ai/DeepSeek-V2-Lite-Chat --enable_skip_router
