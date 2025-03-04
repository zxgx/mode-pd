
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ALLOW_CODE_EVAL="1"

python modepd/eval.py --hf_model /mnt/workspace/zhgeng/models/Qwen2.5-3B-Instruct \
    --tasks humaneval --num_fewshots 0 \
    --batch_size 64 --trust_remote_code --output_dir exp/baseline

python modepd/eval.py --hf_model /mnt/workspace/zhgeng/models/Qwen2.5-3B \
    --batch_size 32 --trust_remote_code --output_dir exp/baseline

python modepd/eval.py --hf_model /mnt/workspace/zhgeng/models/gemma-2-2b-it \
    --tasks humaneval --num_fewshots 0 \
    --batch_size 16 --trust_remote_code --output_dir exp/baseline

python modepd/eval.py --hf_model /mnt/workspace/zhgeng/models/gemma-2-2b \
    --batch_size 8 --trust_remote_code --output_dir exp/baseline

python modepd/eval.py --hf_model /mnt/workspace/zhgeng/DeepSeek-V2-Lite-Chat \
    --tasks humaneval cmmlu --num_fewshots 0 5 \
    --batch_size 8 --trust_remote_code --output_dir exp/baseline
