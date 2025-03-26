#!/bin/bash
#PBS -P CFP01-CF-076
#PBS -l select=1:ngpus=1
#PBS -l place=vscatter
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o debug_expert_intermediate.log

module load openmpi
module load singularity

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


export PYTHONPATH=$PWD:$PYTHONPATH

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

for preserve_n in 48 32; do

    echo "Running expert pruning (intermediate model) with preserve_n: $preserve_n"

    mpirun --hostfile $HOSTFILE --np $NNODES -N 1 \
        singularity exec --bind $SCRATCH:$SCRATCH --nv $image \
        /bin/bash -c "\
        source $SCRATCH/venvs/modepd/bin/activate && \
        python modepd/prune.py \
        --model_name_or_path deepseek-ai/DeepSeek-V2-Lite-Chat\
        --dataset_name_or_path Zyphra/Zyda-2 \
        --dataset_config_name sample-100BT \
        --streaming_dataset \
        --expert_prune \
        --expert_prune_metric mone \
        --preserve_n_experts 48 \
        --expert_ranking_scope model \
        --mone_ranking_metric intermediate_fluctuation\
        --compressed_model_save_path $SCRATCH/DeepSeek-V2-Lite-Chat-Compressed/expert_prune_intermediate_model_${preserve_n} &&\
        python modepd/eval.py\
        --hf_model $SCRATCH/DeepSeek-V2-Lite-Chat-Compressed/expert_prune_intermediate_model_${preserve_n}\
        --tasks ai2_arc boolq copa mmlu openbookqa piqa rte winogrande \
        --num_fewshots 0 0 0 0 0 0 0 0 \
        --trust_remote_code \
        --output_dir $SCRATCH/logs > expert_prune_intermediate_model_${preserve_n}.log 2>&1"

done

rm $HOSTFILE