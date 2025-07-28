#!/bin/bash
#PBS -P CFP02-CF-004
#PBS -l select=1:ngpus=1
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -k oed
#PBS -o benchmark-origin-s1998.log 

cd $PBS_O_WORKDIR; 
echo "JOB ID: $PBS_JOBID, pwd: $PWD, pbs workdir: $PBS_O_WORKDIR"

export PYTHONPATH=$PWD:$PYTHONPATH
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real
export CUDA_VISIBLE_DEVICES=0

image="/scratch/e1154485/images/pytorch_25.03-py3.sif" 
module load singularity

singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/dply/bin/activate

python bench_one_batch.py --model-path Qwen/Qwen3-30B-A3B \
    --batch 1 32 128 512 --input-len 512 --output-len 128 256 512 \
    --impl transformers --disable-cuda-graph --random-seed 1998 \
    --result-filename benchmark-s1998-origin.jsonl
EOF
# --profile --profile-filename-prefix origin
