#!/bin/bash
#PBS -P CFP02-CF-004
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -o freq-dsv2.log 

cd $PBS_O_WORKDIR; 
echo "JOB ID: $PBS_JOBID, pwd: $PWD, pbs workdir: $PBS_O_WORKDIR"

export PYTHONPATH=$PWD:$PYTHONPATH

image="/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif" 
module load singularity

export model_id="DeepSeek-V2-Lite"

for dataset in "zyda2" "c4"; do
    if [ "$dataset" == "c4" ]; then
        . scripts/expert_prune/data_config_c4.sh
    else
        . scripts/expert_prune/data_config.sh
    fi

    for sample_size in 100 500 1000; do

singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
python modepd/prune.py --model_name_or_path deepseek-ai/$model_id $data_config \
    --expert_prune --preserve_n_experts 48 --expert_ranking_scope layer \
    --expert_prune_metric mone --mone_ranking_metric routing_score \
    --zero_out_novice --max_steps $sample_size \
    --compressed_model_save_path iclr_rebuttal/variant_ablation_freq/$dataset-$sample_size/$model_id-mone-pruned-48
EOF

singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
python modepd/prune.py --model_name_or_path deepseek-ai/$model_id $data_config \
    --expert_prune --preserve_n_experts 32 --expert_ranking_scope layer \
    --expert_prune_metric mone --mone_ranking_metric routing_score \
    --zero_out_novice --max_steps $sample_size \
    --compressed_model_save_path iclr_rebuttal/variant_ablation_freq/$dataset-$sample_size/$model_id-mone-pruned-32
EOF

    done
done
