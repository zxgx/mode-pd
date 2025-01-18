#!/bin/bash
#PBS -l select=4:ngpus=4
#PBS -l place=vscatter
#PBS -l walltime=4:00:00
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

module load openmpi/4.1.2-hpe
module load singularity

cd $PBS_O_WORKDIR
echo "JOB ID: $PBS_JOBID, pwd: $PWD, pbs workdir: $PBS_O_WORKDIR"

# for torch.distributed
export NNODES=4
# export NODE_RANK=0
export GPUS_PER_NODE=4
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

# export NCCL_DEBUG=WARN
# export CUDA_LAUNCH_BLOCKING=1

# =============== launch cmds ================
image=${SCRATCH}/images/cuda_12.4.1-cudnn-devel-u22.04.sif

mpirun --hostfile $HOSTFILE --np $NNODES -N 1 \
    singularity exec --bind $SCRATCH:$SCRATCH --nv $image \
    /bin/bash -c "source $SCRATCH/venvs/modepd/bin/activate && \
    export PYTHONPATH=$PBS_O_WORKDIR:\$PYTHONPATH && \
    torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=\${OMPI_COMM_WORLD_RANK} \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    modepd/train.py \
    --output_dir deepseek_v2_10B_finetune\
    --model_name_or_path deepseek-ai/DeepSeek-V2-Lite-Chat \
    --with_tracking \
    --max_train_steps 819 > eval.log 2>&1
"

# torchrun --standalone --nproc_per_node 2 modepd/train.py --with_tracking --output_dir logs/demo
rm $HOSTFILE
