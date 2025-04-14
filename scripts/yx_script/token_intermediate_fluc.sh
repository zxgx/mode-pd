#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH
export HF_DATASETS_TRUST_REMOTE_CODE=1

for preserve_n in 48 32; do
    for expert_ranking_scope in model layer; do
        for mone_ranking_metric in token_fluctuation intermediate_fluctuation; do
            echo "deepseek-ai/DeepSeek-V2-Lite"
            echo "Running expert pruning (${expert_ranking_scope} ${mone_ranking_metric}) with preserve_n: $preserve_n"

            python modepd/prune.py \
                --model_name_or_path deepseek-ai/DeepSeek-V2-Lite\
                --dataset_name_or_path Zyphra/Zyda-2 \
                --dataset_config_name sample-100BT \
                --streaming_dataset \
                --expert_prune \
                --expert_prune_metric mone \
                --preserve_n_experts $preserve_n \
                --expert_ranking_scope $expert_ranking_scope \
                --mone_ranking_metric $mone_ranking_metric\
                --compressed_model_save_path $SCRATCH/DeepSeek-V2/mone_${expert_ranking_scope}_${mone_ranking_metric}_${preserve_n}
            python modepd/eval.py\
                --hf_model $SCRATCH/DeepSeek-V2/mone_${expert_ranking_scope}_${mone_ranking_metric}_${preserve_n}\
                --tasks ai2_arc boolq copa mmlu openbookqa piqa rte winogrande \
                --batch_size 1 \
                --num_fewshots 0 0 0 0 0 0 0 0 \
                --trust_remote_code \
                --output_dir $SCRATCH/logs
        done
    done
done