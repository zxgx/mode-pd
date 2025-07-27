#!/bin/bash
#PBS -P CFP02-CF-004
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -o flap-prune-q2.log 

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
    --weight_prune --weight_prune_metric flap --preserve_channels_in_percent 0.75 \
    --compressed_model_save_path exp/flap/$model_id-flap-pruned-75
EOF

singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
python modepd/prune.py --model_name_or_path Qwen/$model_id $data_config \
    --weight_prune --weight_prune_metric flap --preserve_channels_in_percent 0.50 \
    --compressed_model_save_path exp/flap/$model_id-flap-pruned-50
EOF
