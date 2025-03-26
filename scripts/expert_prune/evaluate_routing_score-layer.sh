#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ALLOW_CODE_EVAL="1"
GPUS_PER_NODE=8

. local_scripts/expert_prune/eval_config.sh

model_id="OLMoE-1B-7B-0125"
torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
    --hf_model exp/routing_score_layer/$model_id-pruned-48 $eval_config \
    --batch_size 8 --trust_remote_code --output_dir exp/routing_score_layer

torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
    --hf_model exp/routing_score_layer/$model_id-pruned-32 $eval_config \
    --batch_size 8 --trust_remote_code --output_dir exp/routing_score_layer

# model_id="Moonlight-16B-A3B-Instruct"
# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model exp/routing_score_layer/$model_id-pruned-48 $eval_config \
#     --batch_size 8 --trust_remote_code --output_dir exp/routing_score_layer

# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model exp/routing_score_layer/$model_id-pruned-32 $eval_config \
#     --batch_size 8 --trust_remote_code --output_dir exp/routing_score_layer

# model_id="DeepSeek-V2-Lite-Chat"
# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model exp/routing_score_layer/$model_id-pruned-48 $eval_config \
#     --batch_size 8 --trust_remote_code --output_dir exp/routing_score_layer

# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model exp/routing_score_layer/$model_id-pruned-32 $eval_config \
#     --batch_size 8 --trust_remote_code --output_dir exp/routing_score_layer
