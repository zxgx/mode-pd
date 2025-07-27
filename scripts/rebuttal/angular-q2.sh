#!/bin/bash
#PBS -P CFP02-CF-004
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -o angular-prune-q2.log 

cd $PBS_O_WORKDIR; 
echo "JOB ID: $PBS_JOBID, pwd: $PWD, pbs workdir: $PBS_O_WORKDIR"

export PYTHONPATH=$PWD:$PYTHONPATH

image="/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif" 
module load singularity

. scripts/expert_prune/data_config.sh

export model_id="Qwen2-57B-A14B"
singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
python modepd/prune.py --model_name_or_path Qwen/$model_id $data_config \
    --layer_prune --layer_prune_metric angular --drop_n_layers 7 \
    --compressed_model_save_path exp/angular/$model_id-angular-pruned-75
EOF

singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
python modepd/prune.py --model_name_or_path Qwen/$model_id $data_config \
    --layer_prune --layer_prune_metric angular --drop_n_layers 14 \
    --compressed_model_save_path exp/angular/$model_id-angular-pruned-50
EOF
