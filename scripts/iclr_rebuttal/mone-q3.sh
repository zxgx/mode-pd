#!/bin/bash
#PBS -P CFP02-CF-004
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -o mone-prune-q3.log 

cd $PBS_O_WORKDIR; 
echo "JOB ID: $PBS_JOBID, pwd: $PWD, pbs workdir: $PBS_O_WORKDIR"

export PYTHONPATH=$PWD:$PYTHONPATH

image="/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif" 
module load singularity

# . scripts/expert_prune/data_config.sh

# export model_id="Qwen3-30B-A3B-Instruct-2507"
# singularity exec --nv $image bash << EOF
# source $HPCTMP/venvs/mone/bin/activate
# python modepd/prune.py --model_name_or_path Qwen/$model_id $data_config \
#     --expert_prune --preserve_n_experts 96 --expert_ranking_scope layer \
#     --expert_prune_metric mone --mone_ranking_metric fusion \
#     --compressed_model_save_path iclr_rebuttal/pruned_instruct_model/$model_id-mone-pruned-96
# EOF

# singularity exec --nv $image bash << EOF
# source $HPCTMP/venvs/mone/bin/activate
# python modepd/prune.py --model_name_or_path Qwen/$model_id $data_config \
#     --expert_prune --preserve_n_experts 64 --expert_ranking_scope layer \
#     --expert_prune_metric mone --mone_ranking_metric fusion \
#     --compressed_model_save_path iclr_rebuttal/pruned_instruct_model/$model_id-mone-pruned-64
# EOF

export model_id="Qwen3-30B-A3B-Instruct-2507"

export data_config="--dataset_name_or_path openai/gsm8k --dataset_config_name main"
singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
python modepd/prune.py --model_name_or_path Qwen/$model_id $data_config \
    --expert_prune --preserve_n_experts 96 --expert_ranking_scope layer \
    --expert_prune_metric mone --mone_ranking_metric fusion \
    --compressed_model_save_path iclr_rebuttal/gsm8k_pruned_instruct_model/$model_id-mone-pruned-96
EOF

export data_config="--dataset_name_or_path EleutherAI/hendrycks_math"
singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
python modepd/prune.py --model_name_or_path Qwen/$model_id $data_config \
    --expert_prune --preserve_n_experts 96 --expert_ranking_scope layer \
    --expert_prune_metric mone --mone_ranking_metric fusion \
    --compressed_model_save_path iclr_rebuttal/math_pruned_instruct_model/$model_id-mone-pruned-96
EOF
