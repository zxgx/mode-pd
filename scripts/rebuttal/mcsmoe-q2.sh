#!/bin/bash
#PBS -P CFP02-CF-004
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -o mcsmoe-prune-q2.log 

cd $PBS_O_WORKDIR; 
echo "JOB ID: $PBS_JOBID, pwd: $PWD, pbs workdir: $PBS_O_WORKDIR"

export PYTHONPATH=$PWD:$PYTHONPATH

image="/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif" 
module load singularity

. scripts/expert_prune/data_config.sh

export model_id="Qwen2-57B-A14B"
# singularity exec --nv $image bash << EOF
# source $HPCTMP/venvs/mone/bin/activate
# python modepd/prune.py --model_name_or_path QWen/$model_id $data_config \
#     --expert_prune --expert_prune_metric mc_smoe --preserve_n_experts 48 \
#     --compressed_model_save_path exp/mc_smoe/$model_id-mc_smoe-pruned-48
# EOF

singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
python modepd/prune.py --model_name_or_path QWen/$model_id $data_config \
    --expert_prune --expert_prune_metric mc_smoe --preserve_n_experts 32 \
    --compressed_model_save_path exp/mc_smoe/$model_id-mc_smoe-pruned-32
EOF
