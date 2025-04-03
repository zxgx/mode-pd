#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH
export HF_DATASETS_TRUST_REMOTE_CODE=1

for prunen in 1 2; do
    echo "deepseek-ai/DeepSeek-V2-Lite"
    echo "Running weight pruning (sparseGPT) with m:n = 4:${prunen}"

    python modepd/prune.py \
        --model_name_or_path deepseek-ai/DeepSeek-V2-Lite\
        --dataset_name_or_path Zyphra/Zyda-2 \
        --dataset_config_name sample-100BT \
        --streaming_dataset \
        --weight_prune \
        --weight_prune_metric sparsegpt \
        --sparsegpt_prunen $prunen \
        --sparsegpt_prunem 4 \
        --compressed_model_save_path $SCRATCH/sparseGpt/DeepSeek-V2-Lite_4_${prunen}
    python modepd/eval.py\
        --hf_model $SCRATCH/sparseGpt/DeepSeek-V2-Lite_4_${prunen}
        --tasks ai2_arc boolq copa mmlu openbookqa piqa rte winogrande \
        --batch_size 1 \
        --num_fewshots 0 0 0 0 0 0 0 0 \
        --trust_remote_code \
        --output_dir $SCRATCH/logs
done