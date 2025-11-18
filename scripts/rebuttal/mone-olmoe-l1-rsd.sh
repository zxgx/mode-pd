#!/bin/bash
#PBS -P CFP02-CF-004
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -o mone-olmoe-l1-rsd.log 

cd $PBS_O_WORKDIR; 
echo "JOB ID: $PBS_JOBID, pwd: $PWD, pbs workdir: $PBS_O_WORKDIR"

export PYTHONPATH=$PWD:$PYTHONPATH

image="/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif" 
module load singularity

. scripts/expert_prune/data_config.sh

export model_id="OLMoE-1B-7B-0125"
singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
python modepd/prune.py --model_name_or_path allenai/$model_id $data_config \
    --expert_prune --preserve_n_experts 48 --expert_ranking_scope layer \
    --expert_prune_metric mone --mone_ranking_metric fusion \
    --compressed_model_save_path exp/ablation/$model_id-mone-pruned-48-l1-rsd \
    --l1_norm --coeff_var

python modepd/prune.py --model_name_or_path allenai/$model_id $data_config \
    --expert_prune --preserve_n_experts 32 --expert_ranking_scope layer \
    --expert_prune_metric mone --mone_ranking_metric fusion \
    --compressed_model_save_path exp/ablation/$model_id-mone-pruned-32-l1-rsd \
    --l1_norm --coeff_var
EOF
