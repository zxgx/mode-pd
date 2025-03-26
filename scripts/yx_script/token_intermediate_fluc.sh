#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH

for preserve_n in 48 32; do
    for expert_ranking_scope in model layer; do
        for mone_ranking_metric in token_fluctuation intermediate_fluctuation; do
            echo "deepseek-ai/DeepSeek-V2-Lite-Chat"
            echo "Running expert pruning (intermediate model) with preserve_n: $preserve_n"

            python modepd/prune.py \
                --model_name_or_path deepseek-ai/DeepSeek-V2-Lite-Chat\
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
                --num_fewshots 0 0 0 0 0 0 0 0 \
                --trust_remote_code \
                --output_dir $SCRATCH/logs
        done
    done
done

for preserve_n in 48 32; do
    for expert_ranking_scope in model layer; do
        for mone_ranking_metric in token_fluctuation intermediate_fluctuation; do
            echo "moonshotai/Moonlight-16B-A3B"
            echo "Running expert pruning (intermediate model) with preserve_n: $preserve_n"

            python modepd/prune.py \
                --model_name_or_path moonshotai/Moonlight-16B-A3B\
                --dataset_name_or_path Zyphra/Zyda-2 \
                --dataset_config_name sample-100BT \
                --streaming_dataset \
                --expert_prune \
                --expert_prune_metric mone \
                --preserve_n_experts $preserve_n \
                --expert_ranking_scope $expert_ranking_scope \
                --mone_ranking_metric $mone_ranking_metric\
                --compressed_model_save_path $SCRATCH/MoonShot/mone_${expert_ranking_scope}_${mone_ranking_metric}_${preserve_n}
            python modepd/eval.py\
                --hf_model $SCRATCH/MoonShot/mone_${expert_ranking_scope}_${mone_ranking_metric}_${preserve_n}\
                --tasks ai2_arc boolq copa mmlu openbookqa piqa rte winogrande \
                --num_fewshots 0 0 0 0 0 0 0 0 \
                --trust_remote_code \
                --output_dir $SCRATCH/logs
        done
    done
done