#!/bin/bash
#PBS -P CFP02-CF-004
#PBS -l select=1:ngpus=8
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -k oed
#PBS -o freq-Moonlight.log 

cd $PBS_O_WORKDIR; 
echo "JOB ID: $PBS_JOBID, pwd: $PWD, pbs workdir: $PBS_O_WORKDIR"

export GPUS_PER_NODE=8
export PYTHONPATH=$PWD:$PYTHONPATH

image="/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif" 
module load singularity

. scripts/iclr_rebuttal/eval_config_general.sh

export model_id="Moonlight-16B-A3B"

for dataset in "c4" "zyda2"; do
    for sample_size in 100 500 1000; do

singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
    --hf_model iclr_rebuttal/variant_ablation_freq/$dataset-$sample_size/$model_id-mone-pruned-48 \
    --tasks $EVAL_TASKS --num_fewshots $EVAL_FEWSHOTS \
    --batch_size 8 --trust_remote_code --output_dir iclr_rebuttal/variant_ablation_freq/$dataset-$sample_size
EOF

singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
    --hf_model iclr_rebuttal/variant_ablation_freq/$dataset-$sample_size/$model_id-mone-pruned-32 \
    --tasks $EVAL_TASKS --num_fewshots $EVAL_FEWSHOTS \
    --batch_size 8 --trust_remote_code --output_dir iclr_rebuttal/variant_ablation_freq/$dataset-$sample_size
EOF

    done
done
