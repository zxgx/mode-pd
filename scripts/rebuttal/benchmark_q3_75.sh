#!/bin/bash
#PBS -P CFP02-CF-004
#PBS -l select=1:ngpus=1
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -k oed
#PBS -o benchmark-q3-75.log 

cd $PBS_O_WORKDIR; 
echo "JOB ID: $PBS_JOBID, pwd: $PWD, pbs workdir: $PBS_O_WORKDIR"

export PYTHONPATH=$PWD:$PYTHONPATH
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real
export CUDA_VISIBLE_DEVICES=0

image="/scratch/e1154485/images/pytorch_25.03-py3.sif" 
module load singularity

export model_id="Qwen3-30B-A3B"
singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/dply/bin/activate

python bench_one_batch.py --model-path exp/mone_rs_out_fusion_layer/$model_id-mone-pruned-96 \
    --batch 1 128 512 --input-len 512 --output-len 256 \
    --impl transformers --disable-cuda-graph --random-seed 42 \
    --result-filename benchmark-s42-q3-75.jsonl
EOF
# --profile --profile-filename-prefix q3-75 