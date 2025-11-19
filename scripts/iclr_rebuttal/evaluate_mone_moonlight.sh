#!/bin/bash
#PBS -P CFP02-CF-004
#PBS -l select=1:ngpus=8
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -k oed
#PBS -o eval-Moonlight.log 

cd $PBS_O_WORKDIR; 
echo "JOB ID: $PBS_JOBID, pwd: $PWD, pbs workdir: $PBS_O_WORKDIR"

export GPUS_PER_NODE=8
export PYTHONPATH=$PWD:$PYTHONPATH

image="/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif" 
module load singularity

# . scripts/iclr_rebuttal/eval_config_math.sh

# export model_id="Moonlight-16B-A3B"
# singularity exec --nv $image bash << EOF
# source $HPCTMP/venvs/mone/bin/activate
# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model exp/ablation/$model_id-mone-pruned-48 \
#     --tasks $EVAL_TASKS --num_fewshots $EVAL_FEWSHOTS \
#     --batch_size 8 --trust_remote_code --output_dir iclr_rebuttal/math
# EOF

# export model_id="Moonlight-16B-A3B-Instruct"
# singularity exec --nv $image bash << EOF
# source $HPCTMP/venvs/mone/bin/activate
# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model iclr_rebuttal/pruned_instruct_model/$model_id-mone-pruned-48 \
#     --tasks $EVAL_TASKS --num_fewshots $EVAL_FEWSHOTS \
#     --batch_size 8 --trust_remote_code --output_dir iclr_rebuttal/math
# EOF

# singularity exec --nv $image bash << EOF
# source $HPCTMP/venvs/mone/bin/activate
# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model iclr_rebuttal/gsm8k_pruned_instruct_model/$model_id-mone-pruned-48 \
#     --tasks $EVAL_TASKS --num_fewshots $EVAL_FEWSHOTS \
#     --batch_size 8 --trust_remote_code --output_dir iclr_rebuttal/gsm8k_pruned_instruct_model
# EOF

# singularity exec --nv $image bash << EOF
# source $HPCTMP/venvs/mone/bin/activate
# torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
#     --hf_model iclr_rebuttal/math_pruned_instruct_model/$model_id-mone-pruned-48 \
#     --tasks $EVAL_TASKS --num_fewshots $EVAL_FEWSHOTS \
#     --batch_size 8 --trust_remote_code --output_dir iclr_rebuttal/math_pruned_instruct_model
# EOF

. scripts/iclr_rebuttal/eval_config_general.sh

export model_id="Moonlight-16B-A3B"

singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
    --hf_model moonshotai/$model_id \
    --tasks $EVAL_TASKS --num_fewshots $EVAL_FEWSHOTS \
    --routing_distribution --stats_output_dir iclr_rebuttal/mone \
    --batch_size 8 --trust_remote_code --output_dir iclr_rebuttal/mone
EOF

singularity exec --nv $image bash << EOF
source $HPCTMP/venvs/mone/bin/activate
torchrun --standalone --nproc_per_node $GPUS_PER_NODE modepd/eval.py \
    --hf_model iclr_rebuttal/mone/$model_id-mone-pruned-48 \
    --tasks $EVAL_TASKS --num_fewshots $EVAL_FEWSHOTS \
    --routing_distribution --stats_output_dir iclr_rebuttal/mone \
    --batch_size 8 --trust_remote_code --output_dir iclr_rebuttal/mone
EOF
