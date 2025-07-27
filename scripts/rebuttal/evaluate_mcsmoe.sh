#!/bin/bash
#PBS -P CFP02-CF-004
#PBS -l select=1:ngpus=8
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -k oed
#PBS -o eval-mcsmoe.log 

cd $PBS_O_WORKDIR; 
echo "JOB ID: $PBS_JOBID, pwd: $PWD, pbs workdir: $PBS_O_WORKDIR"

export GPUS_PER_NODE=8
export PYTHONPATH=$PWD:$PYTHONPATH

image="/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif" 
module load singularity

. scripts/expert_prune/eval_config.sh

export model_id="Qwen3-30B-A3B"
singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
    --hf_model exp/mc_smoe/$model_id-mc_smoe-pruned-96 \
    --tasks $EVAL_TASKS --num_fewshots $EVAL_FEWSHOTS \
    --batch_size 8 --trust_remote_code --output_dir exp/mc_smoe
EOF

singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
    --hf_model exp/mc_smoe/$model_id-mc_smoe-pruned-64 \
    --tasks $EVAL_TASKS --num_fewshots $EVAL_FEWSHOTS \
    --batch_size 8 --trust_remote_code --output_dir exp/mc_smoe
EOF

export model_id="Qwen2-57B-A14B"
singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
    --hf_model exp/mc_smoe/$model_id-mc_smoe-pruned-48 \
    --tasks $EVAL_TASKS --num_fewshots $EVAL_FEWSHOTS \
    --batch_size 8 --trust_remote_code --output_dir exp/mc_smoe
EOF

singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
    --hf_model exp/mc_smoe/$model_id-mc_smoe-pruned-32 \
    --tasks $EVAL_TASKS --num_fewshots $EVAL_FEWSHOTS \
    --batch_size 8 --trust_remote_code --output_dir exp/mc_smoe
EOF