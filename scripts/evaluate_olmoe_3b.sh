#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ALLOW_CODE_EVAL="1"
GPUS_PER_NODE=8

# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model exp/mone_rs_out_fusion_layer/OLMoE-1B-7B-0125-mone-pruned-32 \
#     --batch_size 8 --trust_remote_code --output_dir exp/pretrain/baseline

# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model exp/finetune/OLMoE-1B-3B-step2560_1e-4 \
#     --batch_size 8 --trust_remote_code --output_dir exp/finetune

# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model exp/finetune/OLMoE-1B-3B-step2560_5e-5 \
#     --batch_size 8 --trust_remote_code --output_dir exp/finetune

# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model exp/finetune/OLMoE-1B-3B-step2560_1e-5 \
#     --batch_size 8 --trust_remote_code --output_dir exp/finetune

# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model exp/finetune/OLMoE-1B-3B-step2560_2e-5 \
#     --batch_size 8 --trust_remote_code --output_dir exp/finetune

# for step in 512 1024 1536 2048; do
#     torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#         --hf_model exp/finetune/OLMoE-1B-3B-step2560_5e-5/step_${step}/model \
#         --batch_size 8 --trust_remote_code --output_dir exp/finetune/OLMoE-1B-3B-step2560_5e-5/step_$step
# done

torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
    --hf_model exp/distill/OLMoE-1B-3B \
    --batch_size 8 --trust_remote_code --output_dir exp/distill
