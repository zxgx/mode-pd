#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ALLOW_CODE_EVAL="1"
GPUS_PER_NODE=8

. local_scripts/expert_prune/eval_config.sh

for subdir in zyda2-500 zyda2-1000 c4-100 c4-500 c4-1000; do
    model_id="OLMoE-1B-7B-0125"
    torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
        --hf_model exp/$subdir/mone_rs_out_fusion_layer/$model_id-mone-pruned-48 $eval_config \
        --batch_size 8 --trust_remote_code --output_dir exp/$subdir/mone_rs_out_fusion_layer

    torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
        --hf_model exp/$subdir/mone_rs_out_fusion_layer/$model_id-mone-pruned-32 $eval_config \
        --batch_size 8 --trust_remote_code --output_dir exp/$subdir/mone_rs_out_fusion_layer

    model_id="Moonlight-16B-A3B"
    torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
        --hf_model exp/$subdir/mone_rs_out_fusion_layer/$model_id-mone-pruned-48 $eval_config \
        --batch_size 8 --trust_remote_code --output_dir exp/$subdir/mone_rs_out_fusion_layer

    torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
        --hf_model exp/$subdir/mone_rs_out_fusion_layer/$model_id-mone-pruned-32 $eval_config \
        --batch_size 8 --trust_remote_code --output_dir exp/$subdir/mone_rs_out_fusion_layer

    model_id="DeepSeek-V2-Lite"
    torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
        --hf_model exp/$subdir/mone_rs_out_fusion_layer/$model_id-mone-pruned-48 $eval_config \
        --batch_size 8 --trust_remote_code --output_dir exp/$subdir/mone_rs_out_fusion_layer

    torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
        --hf_model exp/$subdir/mone_rs_out_fusion_layer/$model_id-mone-pruned-32 $eval_config \
        --batch_size 8 --trust_remote_code --output_dir exp/$subdir/mone_rs_out_fusion_layer
done
