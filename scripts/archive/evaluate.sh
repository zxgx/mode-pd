#!/bin/bash
#PBS -P CFP01-CF-076
#PBS -l select=1:ngpus=1
#PBS -l place=vscatter
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o debug.log

# =============== env params ================
# This script is for NSCC which uses PBS Pro as the scheduler

# where the singularity image is saved
# $SCRATCH

# $TMPDIR set by PBS will intervene triton compilation inside singularity
# export RECORD=$TMPDIR
# unset TMPDIR
# echo "JOB TMPDIR: $TMPDIR, record tmpdir: $RECORD"

# Hopper cluster has a cap on this env var up to 64
# export OMP_NUM_THREADS=1

cd $PBS_O_WORKDIR
echo "JOB ID: $PBS_JOBID, pwd: $PWD, pbs workdir: $PBS_O_WORKDIR"

# for torch.distributed
export NNODES=1
# export NODE_RANK=0
export GPUS_PER_NODE=1
export WORLD_SIZE=$(($NNODES*$GPUS_PER_NODE))
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE | awk -F'.' '{print $1}')
export MASTER_PORT=29501
echo "master node: $MASTER_ADDR"

# used by OpenMPI
export HOSTFILE="$PBS_JOBID.hostfile"
# cat $PBS_NODEFILE | awk -F'.' '{for(i=1;i<=NF;i+=6) print $1 " slots="ENVIRON["GPUS_PER_NODE"]}' > $HOSTFILE
cat $PBS_NODEFILE | awk -F'.' '{for(i=1;i<=NF;i+=6) print $1 " slots=1"}' > $HOSTFILE
echo "detected hosts: $(cat $HOSTFILE)"

# refer to: https://apptainer.org/user-docs/master/gpu.html
# for apptainer, replace SINGULARITYENV_* with APPTAINERENV_*
# export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$(printf "%s," $(seq 0 $(($GPUS_PER_NODE-1))) | sed 's/,$//')
# echo "singularity cuda visible devices: $SINGULARITYENV_CUDA_VISIBLE_DEVICES"

# =============== launch cmds ================
image=${SCRATCH}/images/cuda_12.4.1-cudnn-devel-u22.04.sif

# --hf_model is the output dir of finetuned/pruned processes or pretrained model name
mpirun --hostfile $HOSTFILE --np $NNODES -N 1 \
    singularity exec --bind $SCRATCH:$SCRATCH --nv $image \
    /bin/bash -c "source $SCRATCH/venvs/modepd/bin/activate && \
    python modepd/eval.py --hf_model logs/teacher \
    --tasks ceval-valid cmmlu --num_fewshots 5 5 \
    --batch_size 1 --trust_remote_code --output_dir logs > eval.log 2>&1
"

rm $HOSTFILE

# singularity exec --nv $image bash << EOF > eval.log 2>&1
# source /hpctmp/e1154485/venvs/modepd/bin/activate

# python main.py --hf_model deepseek-ai/DeepSeek-V2-Lite \
#     --tasks ceval-valid cmmlu --num_fewshots 5 5 \
#     --batch_size 32 --trust_remote_code --output_dir logs

# python main.py --hf_model deepseek-ai/DeepSeek-V2-Lite-Chat \
#     --tasks ceval-valid cmmlu --num_fewshots 5 5 \
#     --batch_size 32 --trust_remote_code --output_dir logs

# python main.py --hf_model Qwen/Qwen2.5-3B \
#     --tasks ceval-valid cmmlu --num_fewshots 5 5 \
#     --batch_size 64 --output_dir logs

# python main.py --hf_model Qwen/Qwen2.5-3B-Instruct \
#     --batch_size 4 --output_dir logs

# NOTE: Too large to fit in memory
# python main.py --hf_model deepseek-ai/DeepSeek-V2.5-1210 \
#     --trust_remote_code --output_dir logs

# echo "START google/gemma-2-2b"
# python main.py --hf_model google/gemma-2-2b \
#     --batch_size 8 --output_dir logs

# echo "START google/gemma-2-2b-it"
# python main.py --hf_model google/gemma-2-2b-it \
#     --batch_size 8 --output_dir logs

# EOF
