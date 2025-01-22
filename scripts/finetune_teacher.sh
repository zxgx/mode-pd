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

# =============== grid search parameters ================
BLOCK_SIZE=1024

TRAINING_TOKEN_B_VALUES=(0.1 0.5 1 2)
WEIGHT_DECAY_VALUES=(0.1 0.01 0.001)
LEARNING_RATE_VALUES=(2.4e-4 1e-4 5e-5)
LR_SCHEDULER_TYPES=("linear" "cosine" "polynomial")
NUM_WARMUP_STEPS_VALUES=(0 100 200)

# =============== grid search ================
for TRAINING_TOKEN_B in "${TRAINING_TOKEN_B_VALUES[@]}"; do
    # calculate steps
    MAX_TRAIN_STEPS= $(echo "scale=0; ($TRAINING_TOKEN_B * 2 ** 30) / $BLOCK_SIZE / $NNODES / $GPUS_PER_NODE" | bc)

    for WEIGHT_DECAY in "${WEIGHT_DECAY_VALUES[@]}"; do
        for LEARNING_RATE in "${LEARNING_RATE_VALUES[@]}"; do
            for LR_SCHEDULER_TYPE in "${LR_SCHEDULER_TYPES[@]}"; do
                for NUM_WARMUP_STEPS in "${NUM_WARMUP_STEPS_VALUES[@]}"; do
                    echo "Running with $TRAINING_TOKEN_B B tokens, weight_decay=$WEIGHT_DECAY, learning_rate=$LEARNING_RATE, lr_scheduler_type=$LR_SCHEDULER_TYPE, num_warmup_steps=$NUM_WARMUP_STEPS"

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
                        --output_dir logs/teacher_${TRAINING_TOKEN_B}_${WEIGHT_DECAY}_${LEARNING_RATE}_${LR_SCHEDULER_TYPE}_${NUM_WARMUP_STEPS} \
                        --model_name_or_path deepseek-ai/DeepSeek-V2-Lite-Chat \
                        --with_tracking \
                        --weight_decay=$WEIGHT_DECAY \
                        --learning_rate=$LEARNING_RATE \
                        --lr_scheduler_type=$LR_SCHEDULER_TYPE \
                        --num_warmup_steps=$NUM_WARMUP_STEPS \
                        --block_size=$BLOCK_SIZE\
                        --max_train_steps=$MAX_TRAIN_STEPS > eval_${TRAINING_TOKEN_B}_${WEIGHT_DECAY}_${LEARNING_RATE}_${LR_SCHEDULER_TYPE}_${NUM_WARMUP_STEPS}.log 2>&1
                    "

                    # for debug purpose
                    # torchrun --standalone --nproc_per_node 2 modepd/train.py --with_tracking --output_dir logs/demo
                done
            done
        done
    done
done

rm $HOSTFILE
