#!/bin/bash
#PBS -P CFP02-CF-004
#PBS -l select=1:ngpus=8
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -k oed
#PBS -o eval-origin-instruct.log 

cd $PBS_O_WORKDIR; 
echo "JOB ID: $PBS_JOBID, pwd: $PWD, pbs workdir: $PBS_O_WORKDIR"

export GPUS_PER_NODE=8
export PYTHONPATH=$PWD:$PYTHONPATH

image="/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif" 
module load singularity

. scripts/iclr_rebuttal/math_eval_config.sh

output_dir="iclr_rebuttal/baseline"

# export model_id="moonshotai/Moonlight-16B-A3B-Instruct"
# singularity exec --nv $image bash << EOF
# source $HPCTMP/venvs/mone/bin/activate
# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model $model_id \
#     --tasks $EVAL_TASKS --num_fewshots $EVAL_FEWSHOTS \
#     --batch_size 8 --trust_remote_code --output_dir $output_dir
# EOF

# export model_id="allenai/OLMoE-1B-7B-0125-Instruct"
# singularity exec --nv $image bash << EOF
# source $HPCTMP/venvs/mone/bin/activate
# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model $model_id \
#     --tasks $EVAL_TASKS --num_fewshots $EVAL_FEWSHOTS \
#     --batch_size 8 --trust_remote_code --output_dir $output_dir
# EOF

export model_id="deepseek-ai/DeepSeek-V2-Lite-Chat"
singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
    --hf_model $model_id \
    --tasks $EVAL_TASKS --num_fewshots $EVAL_FEWSHOTS \
    --batch_size 8 --trust_remote_code --output_dir $output_dir
EOF

# export model_id="Qwen/Qwen3-30B-A3B-Instruct-2507"
# singularity exec --nv $image bash << EOF
# source $HPCTMP/venvs/mone/bin/activate
# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model $model_id \
#     --tasks $EVAL_TASKS --num_fewshots $EVAL_FEWSHOTS \
#     --batch_size 8 --trust_remote_code --output_dir $output_dir
# EOF

export model_id="Qwen/Qwen2-57B-A14B-Instruct"
singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
    --hf_model $model_id \
    --tasks $EVAL_TASKS --num_fewshots $EVAL_FEWSHOTS \
    --batch_size 8 --trust_remote_code --output_dir $output_dir
EOF
